"""
Script for performing Probabilistic inference using MC Dropout and testing the ood detection
"""
import numpy as np

import core
import os
import sys
import torch
from shutil import copyfile

from inference.baselines import save_energy_scores_baselines

# This is very ugly. Essential for now but should be fixed.
sys.path.append(os.path.join(core.top_dir(), "src", "detr"))

# Detectron imports
from detectron2.engine import launch

# Project imports
# from core.evaluation_tools.evaluation_utils import get_train_contiguous_id_to_test_thing_dataset_id_dict
from core.setup import setup_config, setup_arg_parser
from inference.inference_utils import get_inference_output_dir, build_predictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from detectron2.data.detection_utils import read_image
# Latent space OOD detection imports
# The following matplotlib backend (TkAgg) seems to be the only one that easily can plot either on the main or in
# the second screen. Remove or change the matplotlib backend if it doesn't work well
# import matplotlib
# matplotlib.use('TkAgg')
from ls_ood_detect.uncertainty_estimation import Hook, RouteDICE, get_predictive_uncertainty_score
from ls_ood_detect.uncertainty_estimation import deeplabv3p_apply_dropout
from ls_ood_detect.uncertainty_estimation import get_dl_h_z
from ls_ood_detect.rcnn import get_ls_mcd_samples_rcnn, get_msp_score_rcnn, get_dice_feat_mean_react_percentile_rcnn
from helper_functions import (
    build_in_distribution_valid_test_dataloader_args,
    build_data_loader,
    build_ood_dataloader_args, build_ind_voc_train_dataloader_args,
)

EXTRACT_MCD_SAMPLES_AND_ENTROPIES = True
EXTRACT_IND = True
EXTRACT_OOD = False
BASELINES = ["pred_h", "mi", "ash", "react", "dice", "dice_react", "msp", "energy"]


def main(args) -> None:
    """
    The current script has as only purpose to get the Monte Carlo Dropout samples, save them,
    and then calculate the entropy and save those quantities for further analysis. This will do this for the InD BDD set
    and one chosen OoD set
    :param args: Configuration class parameters
    :return: None
    """
    # Setup config
    cfg = setup_config(args, random_seed=args.random_seed, is_testing=True)
    # Make sure only 1 data point is processed at a time. This simulates
    # deployment.
    cfg.defrost()
    # cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = 1

    cfg.MODEL.DEVICE = device.type
    # Set up number of cpu threads#
    # torch.set_num_threads(cfg.DATALOADER.NUM_WORKERS)

    # Create inference output directory and copy inference config file to keep
    # track of experimental settings
    inference_output_dir = get_inference_output_dir(
        cfg["OUTPUT_DIR"],
        args.test_dataset,
        args.inference_config,
        args.image_corruption_level,
    )
    ind_dataset = args.config_file.split("-Detection")[0][-3:].lower()
    assert ind_dataset in ("bdd", "voc")
    assert cfg.PROBABILISTIC_INFERENCE.OOD_DATASET in (
        "coco_ood_val_bdd",
        "openimages_ood_val",
        "coco_ood_val",
        "bdd_custom_val"
    )

    os.makedirs(inference_output_dir, exist_ok=True)
    copyfile(
        args.inference_config,
        os.path.join(inference_output_dir, os.path.split(args.inference_config)[-1]),
    )
    # Samples save folder
    SAVE_FOLDER = f"./MCD_evaluation_data/{cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.LAYER_TYPE}/"
    # Assert only one layer is specified to be hooked
    assert (
            cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.HOOK_RELU_AFTER_DROPOUT
            + cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.HOOK_DROPOUT_BEFORE_RELU
            + cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.HOOK_DROPBLOCK_RPN
            + cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.HOOK_DROPBLOCK_AFTER_BACKBONE
            == 1
    ), " Select only one layer to be hooked"
    ##################################################################################
    # Prepare predictor and data loaders
    ##################################################################################
    # Build predictor
    predictor = build_predictor(cfg)
    if cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.HOOK_RELU_AFTER_DROPOUT:
        # Hook the final activation of the module: the ReLU after the dropout
        hooked_dropout_layer = Hook(predictor.model.roi_heads.box_head)
    elif cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.HOOK_DROPOUT_BEFORE_RELU:
        # Place the Hook at the output of the last dropout layer
        hooked_dropout_layer = Hook(predictor.model.roi_heads.box_head.fc_dropout2)
    elif cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.HOOK_DROPBLOCK_RPN:
        hooked_dropout_layer = Hook(
            predictor.model.proposal_generator.rpn_head.dropblock
        )
    elif cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.HOOK_DROPBLOCK_AFTER_BACKBONE:
        hooked_dropout_layer = Hook(
            predictor.model.backbone
        )
    # Put model in evaluation mode
    predictor.model.eval()
    # Activate Dropout layers
    predictor.model.apply(deeplabv3p_apply_dropout)
    # Build In Distribution valid and test data loader
    if ind_dataset == "voc" or args.test_dataset == "voc_custom_val":
        # The following split proportions correspond to: 0.5 if ind is VOC (and test is also VOC)
        # and to 0.01 when BDD is InD and VOC is OoD (test dataset voc)
        train_voc_args = build_ind_voc_train_dataloader_args(cfg=cfg,
                                                             split_proportion=0.5 if ind_dataset == "voc" else 0.01)
        (
            ind_test_dl_args,
            _,
        ) = build_in_distribution_valid_test_dataloader_args(
            cfg, dataset_name=args.test_dataset, split_proportion=0.41
        )
        ind_dataset_dict = {
            "valid": build_data_loader(**train_voc_args),
            "test": build_data_loader(**ind_test_dl_args)
        }
        del ind_test_dl_args
        del train_voc_args
    # BDD
    else:
        (
            ind_valid_dl_args,
            ind_test_dl_args,
        ) = build_in_distribution_valid_test_dataloader_args(
            cfg, dataset_name=args.test_dataset, split_proportion=0.8
        )
        ind_dataset_dict = {
            "valid": build_data_loader(**ind_valid_dl_args),
            "test": build_data_loader(**ind_test_dl_args)
        }
        del ind_valid_dl_args
        del ind_test_dl_args

    # Build Out of Distribution test data loader
    ood_data_loader_args = build_ood_dataloader_args(cfg)
    ood_test_data_loader = build_data_loader(**ood_data_loader_args)
    if cfg.PROBABILISTIC_INFERENCE.OOD_DATASET == "coco_ood_val_bdd":
        ood_ds_name = "coco"
    elif cfg.PROBABILISTIC_INFERENCE.OOD_DATASET == "coco_ood_val":
        ood_ds_name = "coco_indvoc"
    else:
        ood_ds_name = "openimages"
    del ood_data_loader_args

    ###################################################################################################
    # Perform MCD inference and save samples
    ###################################################################################################
    if EXTRACT_MCD_SAMPLES_AND_ENTROPIES:
        if EXTRACT_IND:
            for split, dataloader in ind_dataset_dict.items():
                # Get Monte-Carlo samples
                ind_samples, ind_raw_samples = get_ls_mcd_samples_rcnn(
                    model=predictor,
                    data_loader=dataloader,
                    mcd_nro_samples=cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.NUM_RUNS,
                    hook_dropout_layer=hooked_dropout_layer,
                    layer_type=cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.LAYER_TYPE,
                    return_raw_predictions=True
                )
                # Save MC samples
                num_images_to_save = int(
                    ind_samples.shape[0] / cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.NUM_RUNS
                )
                torch.save(
                    ind_samples,
                    f"./{SAVE_FOLDER}/{ind_dataset}_{split}_{cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.LAYER_TYPE}_"
                    f"{num_images_to_save}_{ind_samples.shape[1]}_{cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.NUM_RUNS}_"
                    f"mcd_samples.pt",
                )
                # Calculate Mutual information and predictive entropy
                if "mi" in BASELINES or "pred_h" in BASELINES:
                    if split == "test":
                        ind_pred_h, ind_mi = get_predictive_uncertainty_score(
                            input_samples=ind_raw_samples,
                            mcd_nro_samples=cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.NUM_RUNS
                        )
                        ind_pred_h, ind_mi = ind_pred_h.cpu().numpy(), ind_mi.cpu().numpy()
                        np.save(f"./{SAVE_FOLDER}/{ind_dataset}_ind_pred_h", ind_pred_h)
                        np.save(f"./{SAVE_FOLDER}/{ind_dataset}_ind_mi", ind_mi)
                        del ind_mi
                        del ind_pred_h
                # Calculate entropy for InD
                _, ind_h_z_np = get_dl_h_z(
                    ind_samples,
                    mcd_samples_nro=cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.NUM_RUNS,
                )
                # Save entropy calculations
                np.save(
                    f"./{SAVE_FOLDER}/{ind_dataset}_{split}_{cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.LAYER_TYPE}_"
                    f"{ind_h_z_np.shape[0]}_{ind_h_z_np.shape[1]}_{cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.NUM_RUNS}_"
                    f"mcd_h_z_samples",
                    ind_h_z_np,
                )
            del ind_h_z_np
            del ind_raw_samples
            del ind_samples

        if EXTRACT_OOD:
            # OoD
            # # Get Monte-Carlo samples
            ood_test_mc_samples, ood_raw_samples = get_ls_mcd_samples_rcnn(
                model=predictor,
                data_loader=ood_test_data_loader,
                mcd_nro_samples=cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.NUM_RUNS,
                hook_dropout_layer=hooked_dropout_layer,
                layer_type=cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.LAYER_TYPE,
                return_raw_predictions=True
            )

            # Save MC samples
            num_images_to_save = int(
                ood_test_mc_samples.shape[0] / cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.NUM_RUNS
            )
            torch.save(
                ood_test_mc_samples,
                f"./{SAVE_FOLDER}/{ood_ds_name}_ood_test_{cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.LAYER_TYPE}_{num_images_to_save}_{ood_test_mc_samples.shape[1]}_{cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.NUM_RUNS}_mcd_samples.pt",
            )
            if "mi" in BASELINES or "pred_h" in BASELINES:
                ood_pred_h, ood_mi = get_predictive_uncertainty_score(
                    input_samples=ood_raw_samples,
                    mcd_nro_samples=cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.NUM_RUNS
                )
                ood_pred_h, ood_mi = ood_pred_h.cpu().numpy(), ood_mi.cpu().numpy()
                np.save(f"./{SAVE_FOLDER}/{ood_ds_name}_ood_pred_h", ood_pred_h)
                np.save(f"./{SAVE_FOLDER}/{ood_ds_name}_ood_mi", ood_mi)

            # Calculate entropy ood test set
            _, ood_h_z_np = get_dl_h_z(
                ood_test_mc_samples,
                mcd_samples_nro=cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.NUM_RUNS,
            )
            # Save entropy calculations
            np.save(
                f"./{SAVE_FOLDER}/{ood_ds_name}_ood_test_{cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.LAYER_TYPE}_{ood_h_z_np.shape[0]}_{ood_h_z_np.shape[1]}_{cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.NUM_RUNS}_mcd_h_z_samples",
                ood_h_z_np,
            )
            # Since inference if memory-intense, we want to liberate as much memory as possible
            del ood_h_z_np
            del ood_pred_h
            del ood_mi
            del ood_test_mc_samples
            del ood_raw_samples

    #######################
    # Maximum softmax probability and energy scores calculations
    predictor.model.eval()  # No MCD needed here
    if "msp" in BASELINES:
        if EXTRACT_IND:
            # InD
            print(f"\nMsp from InD {ind_dataset}")
            assert cfg.PROBABILISTIC_INFERENCE.OUTPUT_BOX_CLS
            ind_test_msp = get_msp_score_rcnn(dnn_model=predictor, input_dataloader=ind_dataset_dict["test"])
            np.save(f"./{SAVE_FOLDER}/{ind_dataset}_ind_msp", ind_test_msp)
        if EXTRACT_OOD:
            # OoD
            print(f"\nMsp from OoD {ood_ds_name}")
            ood_test_msp = get_msp_score_rcnn(dnn_model=predictor, input_dataloader=ood_test_data_loader)
            np.save(f"./{SAVE_FOLDER}/{ood_ds_name}_ood_msp", ood_test_msp)
    if "energy" in BASELINES:
        assert cfg.PROBABILISTIC_INFERENCE.OUTPUT_BOX_CLS
        if EXTRACT_IND:
            save_energy_scores_baselines(predictor=predictor,
                                         data_loader=ind_dataset_dict["test"],
                                         baseline_name="energy",
                                         save_foldar_name=SAVE_FOLDER,
                                         ds_name=ind_dataset,
                                         ds_type="ind")
        if EXTRACT_OOD:
            save_energy_scores_baselines(predictor=predictor,
                                         data_loader=ood_test_data_loader,
                                         baseline_name="energy",
                                         save_foldar_name=SAVE_FOLDER,
                                         ds_name=ood_ds_name,
                                         ds_type="ood")

    ##########################
    # ASH
    if "ash" in BASELINES:
        predictor.ash_inference = True
        if EXTRACT_IND:
            save_energy_scores_baselines(predictor=predictor,
                                         data_loader=ind_dataset_dict["test"],
                                         baseline_name="ash",
                                         save_foldar_name=SAVE_FOLDER,
                                         ds_name=ind_dataset,
                                         ds_type="ind")
        if EXTRACT_OOD:
            save_energy_scores_baselines(predictor=predictor,
                                         data_loader=ood_test_data_loader,
                                         baseline_name="ash",
                                         save_foldar_name=SAVE_FOLDER,
                                         ds_name=ood_ds_name,
                                         ds_type="ood")
        predictor.ash_inference = False

    ###############################
    # DICE, ReAct
    if "dice" in BASELINES or "react" in BASELINES or "dice_react" in BASELINES:
        if not os.path.exists(f"./{SAVE_FOLDER}/dice_info.npy") \
                or not os.path.exists(f"./{SAVE_FOLDER}/react_threshold.npy"):
            predictor.dice_react_precompute = True
            dice_info_mean, react_threshold = get_dice_feat_mean_react_percentile_rcnn(
                dnn_model=predictor,
                ind_dataloader=ind_dataset_dict["valid"],
                react_percentile=cfg.PROBABILISTIC_INFERENCE.REACT_PERCENTILE
            )
            np.save(f"./{SAVE_FOLDER}/dice_info", dice_info_mean)
            np.save(f"./{SAVE_FOLDER}/react_threshold", react_threshold)
            predictor.dice_react_precompute = False
        else:
            dice_info_mean = np.load(file=f"./{SAVE_FOLDER}/dice_info.npy")
            react_threshold = float(np.load(file=f"./{SAVE_FOLDER}/react_threshold.npy"))
        # react_threshold = 0.8  # Only for debugging purposes
        # dice_info_mean = np.random.rand(1024)  # Only for debugging purposes
        if "react" in BASELINES:
            # React evaluation
            predictor.react_threshold = react_threshold
            if EXTRACT_IND:
                save_energy_scores_baselines(predictor=predictor,
                                             data_loader=ind_dataset_dict["test"],
                                             baseline_name="react",
                                             save_foldar_name=SAVE_FOLDER,
                                             ds_name=ind_dataset,
                                             ds_type="ind")
            if EXTRACT_OOD:
                save_energy_scores_baselines(predictor=predictor,
                                             data_loader=ood_test_data_loader,
                                             baseline_name="react",
                                             save_foldar_name=SAVE_FOLDER,
                                             ds_name=ood_ds_name,
                                             ds_type="ood")
            predictor.react_threshold = None
        if "dice" in BASELINES:
            # DICE evaluation
            predictor.model.roi_heads.box_predictor.cls_score = RouteDICE(1024,
                                                                          cfg.MODEL.ROI_HEADS.NUM_CLASSES + 1,
                                                                          bias=True,
                                                                          p=cfg.PROBABILISTIC_INFERENCE.DICE_PERCENTILE,
                                                                          info=dice_info_mean).to(device)
            if EXTRACT_IND:
                save_energy_scores_baselines(predictor=predictor,
                                             data_loader=ind_dataset_dict["test"],
                                             baseline_name="dice",
                                             save_foldar_name=SAVE_FOLDER,
                                             ds_name=ind_dataset,
                                             ds_type="ind")
            if EXTRACT_OOD:
                save_energy_scores_baselines(predictor=predictor,
                                             data_loader=ood_test_data_loader,
                                             baseline_name="dice",
                                             save_foldar_name=SAVE_FOLDER,
                                             ds_name=ood_ds_name,
                                             ds_type="ood")
            # Restore model to original
            predictor = build_predictor(cfg)
            predictor.model.eval()

        if "dice_react" in BASELINES:
            # DICE + ReAct evaluation
            predictor.react_threshold = react_threshold
            predictor.model.roi_heads.box_predictor.cls_score = RouteDICE(1024,
                                                                          cfg.MODEL.ROI_HEADS.NUM_CLASSES + 1,
                                                                          bias=True,
                                                                          p=cfg.PROBABILISTIC_INFERENCE.DICE_PERCENTILE,
                                                                          info=dice_info_mean).to(device)
            if EXTRACT_IND:
                save_energy_scores_baselines(predictor=predictor,
                                             data_loader=ind_dataset_dict["test"],
                                             baseline_name="dice_react",
                                             save_foldar_name=SAVE_FOLDER,
                                             ds_name=ind_dataset,
                                             ds_type="ind")
            if EXTRACT_OOD:
                save_energy_scores_baselines(predictor=predictor,
                                             data_loader=ood_test_data_loader,
                                             baseline_name="dice_react",
                                             save_foldar_name=SAVE_FOLDER,
                                             ds_name=ood_ds_name,
                                             ds_type="ood")
            # Restore model to original
            predictor = build_predictor(cfg)
            predictor.model.eval()

    # Analysis of the calculated samples is performed in another script!


if __name__ == "__main__":
    # Create arg parser
    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args()

    print("Command Line Args:", args)
    # This function checks if there are multiple gpus, then it launches the distributed inference, otherwise it
    # just launches the main function, i.e., would act as a function wrapper passing the args to main
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
