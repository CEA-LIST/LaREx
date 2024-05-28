import numpy as np
import os
import torch
import hydra
from omegaconf import DictConfig
from datasets.custom_dataloaders import get_data_loaders_image_classification
from models import ResnetModule
from dropblock import DropBlock2D
from ls_ood_detect.uncertainty_estimation import Hook, MCDSamplesExtractor, \
    get_msp_score, get_energy_score, MDSPostprocessor, KNNPostprocessor, get_dice_feat_mean_react_percentile
from ls_ood_detect.uncertainty_estimation import get_dl_h_z
import warnings

# Datasets paths
datasets_paths_dict = {
    "gtsrb": "./data/gtsrb-data/",
    "cifar10": "./data/cifar10-data/",
    "stl10": "./data/stl10-data/",
    "fmnist": "./data/fmnist-data",
    "svhn": './data/svhn-data/',
    "places": "./data/places-data",
    "textures": "./data/textures-data",
    "lsun_c": "./data/LSUN",
    "lsun_r": "./data/LSUN_resize",
    "isun": "./data/iSUN"
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_WORKERS = os.cpu_count() - 4 if os.cpu_count() >= 8 else os.cpu_count() - 2
EXTRACT_MCD_SAMPLES_AND_ENTROPIES = True  # set False for debugging purposes
EXTRACT_IND = True


@hydra.main(version_base=None, config_path="configs", config_name="config.yaml")
def extract_and_save_mcd_samples(cfg: DictConfig) -> None:
    assert cfg.model.spectral_norm_only_fc + cfg.model.spectral_norm <= 1
    ################################################################################################
    #                                 LOAD DATASETS                                   ##############
    ################################################################################################
    ind_dataset_dict, ood_datasets_dict = get_data_loaders_image_classification(
        cfg=cfg,
        datasets_paths=datasets_paths_dict,
        n_workers=N_WORKERS
    )

    ####################################################################
    # Load trained model
    ####################################################################
    rn_model = ResnetModule.load_from_checkpoint(checkpoint_path=cfg.model_path)
    rn_model.eval()

    # Add Hooks
    if cfg.layer_type == "FC":
        hooked_layer = Hook(rn_model.model.dropout_layer)
    # Conv (dropblock)
    else:
        hooked_layer = Hook(rn_model.model.dropblock2d_layer)

    # Monte Carlo Dropout - Enable Dropout @ Test Time!
    def resnet18_enable_dropblock2d_test(m):
        if type(m) == DropBlock2D or type(m) == torch.nn.Dropout:
            m.train()

    rn_model.to(device)
    rn_model.eval()
    rn_model.apply(resnet18_enable_dropblock2d_test)  # enable dropout

    # Create data folder with the name of the model if it doesn't exist
    mcd_samples_folder = f"./Mcd_samples/ind_{cfg.ind_dataset}/"
    os.makedirs(mcd_samples_folder, exist_ok=True)
    save_dir = f"{mcd_samples_folder}{cfg.model_path.split('/')[2]}/{cfg.layer_type}"
    if os.path.exists(save_dir):
        warnings.warn(f"Destination folder {save_dir} already exists!")
    os.makedirs(save_dir, exist_ok=True)
    ####################################################################################################################
    ####################################################################################################################
    if EXTRACT_MCD_SAMPLES_AND_ENTROPIES:
        #########################################################################
        # Extract MCDO latent samples
        #########################################################################
        # Extract MCD samples
        mcd_extractor = MCDSamplesExtractor(
            model=rn_model.model,
            mcd_nro_samples=cfg.mcd_n_samples,
            hook_dropout_layer=hooked_layer,
            layer_type=cfg.layer_type,
            device=device,
            architecture=cfg.architecture,
            location=cfg.hook_location,
            reduction_method=cfg.reduction_method,
            input_size=cfg.datamodule.image_width,
            original_resnet_architecture=cfg.original_resnet_architecture,
            return_raw_predictions=True
        )
        if EXTRACT_IND:
            # Extract and save InD samples and entropies
            ind_valid_test_entropies = []
            for split, data_loader in ind_dataset_dict.items():
                print(f"\nExtracting InD {cfg.ind_dataset} {split}")
                mcd_samples, mcd_preds = mcd_extractor.get_ls_mcd_samples_baselines(data_loader)
                torch.save(
                    mcd_samples,
                    f"{save_dir}/{cfg.ind_dataset}_{split}_{mcd_samples.shape[0]}_{mcd_samples.shape[1]}_mcd_samples.pt",
                )
                if not split == "train":
                    torch.save(
                        mcd_preds,
                        f"{save_dir}/{cfg.ind_dataset}_{split}_mcd_preds.pt",
                    )
                ind_h_z_samples_np = get_dl_h_z(mcd_samples,
                                                mcd_samples_nro=cfg.mcd_n_samples,
                                                parallel_run=True)[1]
                if split == "train":
                    np.save(
                        f"{save_dir}/{cfg.ind_dataset}_h_z_train", ind_h_z_samples_np,
                    )
                else:
                    ind_valid_test_entropies.append(ind_h_z_samples_np)

            ind_h_z = np.concatenate(
                (ind_valid_test_entropies[0], ind_valid_test_entropies[1])
            )
            np.save(
                f"{save_dir}/{cfg.ind_dataset}_h_z", ind_h_z,
            )
            del mcd_samples
            del mcd_preds
            del ind_h_z
            del ind_valid_test_entropies
            del ind_h_z_samples_np

        # Extract and save OoD samples and entropies
        for dataset_name, data_loaders in ood_datasets_dict.items():
            print(f"Saving samples and entropies from {dataset_name}")
            mcd_samples_ood_v, mcd_preds_ood_v = mcd_extractor.get_ls_mcd_samples_baselines(data_loaders["valid"])
            torch.save(
                mcd_samples_ood_v,
                f"{save_dir}/{dataset_name}_valid_{mcd_samples_ood_v.shape[0]}_{mcd_samples_ood_v.shape[1]}_mcd_samples.pt",
            )
            torch.save(mcd_preds_ood_v, f"{save_dir}/{dataset_name}_valid_mcd_preds.pt")
            mcd_samples_ood_t, mcd_preds_ood_t = mcd_extractor.get_ls_mcd_samples_baselines(data_loaders["test"])
            torch.save(
                mcd_samples_ood_t,
                f"{save_dir}/{dataset_name}_test_{mcd_samples_ood_t.shape[0]}_{mcd_samples_ood_t.shape[1]}_mcd_samples.pt",
            )
            torch.save(mcd_preds_ood_t, f"{save_dir}/{dataset_name}_test_mcd_preds.pt")
            h_z_valid_np = get_dl_h_z(mcd_samples_ood_v,
                                      mcd_samples_nro=cfg.mcd_n_samples,
                                      parallel_run=True)[1]
            h_z_test_np = get_dl_h_z(mcd_samples_ood_t,
                                     mcd_samples_nro=cfg.mcd_n_samples,
                                     parallel_run=True)[1]
            ood_h_z = np.concatenate((h_z_valid_np, h_z_test_np))
            np.save(f"{save_dir}/{dataset_name}_h_z", ood_h_z)

        del mcd_preds_ood_v
        del mcd_preds_ood_t
        del ood_h_z
        del h_z_test_np
        del h_z_valid_np
        del mcd_samples_ood_t
        del mcd_samples_ood_v

    #######################
    # Maximum softmax probability and energy scores calculations
    rn_model.eval()  # No MCD needed here
    if EXTRACT_IND:
        # InD data
        ind_valid_test_msp = []
        ind_valid_test_energy = []
        for split, data_loader in ind_dataset_dict.items():
            if not split == "train":
                print(f"\nMsp and energy from InD {split}")
                if "msp" in cfg.baselines:
                    ind_valid_test_msp.append(
                        get_msp_score(dnn_model=rn_model.model, input_dataloader=data_loader)
                    )
                if "energy" in cfg.baselines:
                    ind_valid_test_energy.append(
                        get_energy_score(dnn_model=rn_model.model, input_dataloader=data_loader)
                    )

        # Concatenate
        if "msp" in cfg.baselines:
            ind_msp_score = np.concatenate((ind_valid_test_msp[0], ind_valid_test_msp[1]))
            np.save(f"{save_dir}/{cfg.ind_dataset}_msp", ind_msp_score)
        if "energy" in cfg.baselines:
            ind_energy_score = np.concatenate((ind_valid_test_energy[0], ind_valid_test_energy[1]))
            np.save(f"{save_dir}/{cfg.ind_dataset}_energy", ind_energy_score)

    for dataset_name, data_loaders in ood_datasets_dict.items():
        print(f"\nmsp and energy from OoD {dataset_name}")
        if "msp" in cfg.baselines:
            ood_valid_msp_score = get_msp_score(dnn_model=rn_model.model, input_dataloader=data_loaders["valid"])
            ood_test_msp_score = get_msp_score(dnn_model=rn_model.model, input_dataloader=data_loaders["test"])
            ood_msp_score = np.concatenate((ood_valid_msp_score, ood_test_msp_score))
            np.save(f"{save_dir}/{dataset_name}_msp", ood_msp_score)
        if "energy" in cfg.baselines:
            ood_test_energy_score = get_energy_score(dnn_model=rn_model.model, input_dataloader=data_loaders["test"])
            ood_valid_energy_score = get_energy_score(dnn_model=rn_model.model, input_dataloader=data_loaders["valid"])
            ood_energy_score = np.concatenate((ood_test_energy_score, ood_valid_energy_score))
            np.save(f"{save_dir}/{dataset_name}_energy", ood_energy_score)

    #############################
    # Prepare Mahalanobis distance and kNN scores estimators
    # Hook now the average pool layer
    gtsrb_model_avgpool_layer_hook = Hook(rn_model.model.avgpool)
    if "mdist" in cfg.baselines:
        # Instantiate and setup estimator
        m_dist_estimator = MDSPostprocessor(num_classes=10 if cfg.ind_dataset == "cifar10" else 43, setup_flag=False)
        m_dist_estimator.setup(rn_model.model, ind_dataset_dict["train"], layer_hook=gtsrb_model_avgpool_layer_hook)
    if "knn" in cfg.baselines:
        # Instantiate kNN postprocessor
        knn_dist_estimator = KNNPostprocessor(K=50, setup_flag=False)
        knn_dist_estimator.setup(rn_model.model, ind_dataset_dict["train"], layer_hook=gtsrb_model_avgpool_layer_hook)

    # Get results from Mahalanobis and kNN estimators
    # InD samples
    ind_valid_test_mdist = []
    ind_valid_test_knn = []
    for split, data_loader in ind_dataset_dict.items():
        if not split == "train":
            print(f"\nMdist and kNN from InD {split}")
            if "mdist" in cfg.baselines:
                ind_valid_test_mdist.append(
                    m_dist_estimator.postprocess(rn_model.model, data_loader, gtsrb_model_avgpool_layer_hook)[1]
                )
            if "knn" in cfg.baselines:
                ind_valid_test_knn.append(
                    knn_dist_estimator.postprocess(rn_model.model, data_loader, gtsrb_model_avgpool_layer_hook)[1]
                )
    if "mdist" in cfg.baselines:
        # Concatenate ind samples
        ind_mdist_score = np.concatenate((ind_valid_test_mdist[0], ind_valid_test_mdist[1]))
        np.save(f"{save_dir}/{cfg.ind_dataset}_mdist", ind_mdist_score)
    if "knn" in cfg.baselines:
        ind_knn_score = np.concatenate((ind_valid_test_knn[0], ind_valid_test_knn[1]))
        np.save(f"{save_dir}/{cfg.ind_dataset}_knn", ind_knn_score)
    if EXTRACT_IND:
        # InD samples
        ind_valid_test_mdist = []
        ind_valid_test_knn = []
        for split, data_loader in ind_dataset_dict.items():
            if not split == "train":
                print(f"\nMdist and kNN from InD {split}")
                if "mdist" in cfg.baselines:
                    ind_valid_test_mdist.append(
                        m_dist_estimator.postprocess(rn_model.model, data_loader, gtsrb_model_avgpool_layer_hook)[1]
                    )
                if "knn" in cfg.baselines:
                    ind_valid_test_knn.append(
                        knn_dist_estimator.postprocess(rn_model.model, data_loader, gtsrb_model_avgpool_layer_hook)[1]
                    )
        if "mdist" in cfg.baselines:
            # Concatenate ind samples
            ind_mdist_score = np.concatenate((ind_valid_test_mdist[0], ind_valid_test_mdist[1]))
            np.save(f"{save_dir}/{cfg.ind_dataset}_mdist", ind_mdist_score)
        if "knn" in cfg.baselines:
            ind_knn_score = np.concatenate((ind_valid_test_knn[0], ind_valid_test_knn[1]))
            np.save(f"{save_dir}/{cfg.ind_dataset}_knn", ind_knn_score)
    # OoD samples
    for dataset_name, data_loaders in ood_datasets_dict.items():
        print(f"\nMdist and kNN from OoD {dataset_name}")
        if "mdist" in cfg.baselines:
            # Mdist
            ood_valid_m_dist_score = m_dist_estimator.postprocess(rn_model.model,
                                                                  data_loaders["valid"],
                                                                  gtsrb_model_avgpool_layer_hook)[1]
            ood_test_m_dist_score = m_dist_estimator.postprocess(rn_model.model,
                                                                 data_loaders["test"],
                                                                 gtsrb_model_avgpool_layer_hook)[1]
            ood_m_dist_score = np.concatenate((ood_valid_m_dist_score, ood_test_m_dist_score))
            np.save(f"{save_dir}/{dataset_name}_mdist", ood_m_dist_score)
        if "knn" in cfg.baselines:
            # kNN
            ood_valid_kth_dist_score = knn_dist_estimator.postprocess(rn_model.model,
                                                                      data_loaders["valid"],
                                                                      gtsrb_model_avgpool_layer_hook)[1]

            ood_test_kth_dist_score = knn_dist_estimator.postprocess(rn_model.model,
                                                                     data_loaders["test"],
                                                                     gtsrb_model_avgpool_layer_hook)[1]
            ood_kth_dist_score = np.concatenate((ood_valid_kth_dist_score, ood_test_kth_dist_score))
            np.save(f"{save_dir}/{dataset_name}_knn", ood_kth_dist_score)

    ##########################################
    # ASH-P
    #######################################
    if "ash" in cfg.baselines:
        rn_model.model.ash = True
        rn_model.eval()  # No MCD needed here
        # InD data
        ind_valid_test_ash = []
        for split, data_loader in ind_dataset_dict.items():
            if not split == "train":
                print(f"\nASH from InD {split}")
                ind_valid_test_ash.append(
                    get_energy_score(dnn_model=rn_model.model, input_dataloader=data_loader)
                )

        # Concatenate and save
        ind_ash_score = np.concatenate((ind_valid_test_ash[0], ind_valid_test_ash[1]))
        np.save(f"{save_dir}/{cfg.ind_dataset}_ash", ind_ash_score)

        for dataset_name, data_loaders in ood_datasets_dict.items():
            print(f"\nASH from OoD {dataset_name}")
            ood_test_ash_score = get_energy_score(dnn_model=rn_model.model, input_dataloader=data_loaders["test"])
            ood_valid_ash_score = get_energy_score(dnn_model=rn_model.model, input_dataloader=data_loaders["valid"])
            ood_ash_score = np.concatenate((ood_test_ash_score, ood_valid_ash_score))
            np.save(f"{save_dir}/{dataset_name}_ash", ood_ash_score)
        rn_model.model.ash = False

    ##########################################
    # DICE and ReAct
    #######################################
    if "dice" in cfg.baselines or "react" in cfg.baselines or "dice_react" in cfg.baselines:
        # Precompute DICE and ReAct threshold
        rn_model.model.dice_precompute = True
        dice_info_mean, react_threshold = get_dice_feat_mean_react_percentile(rn_model.model,
                                                                              ind_dataset_dict["train"],
                                                                              cfg.react_percentile)
        #######
        # ReAct
        if "react" in cfg.baselines:
            # Inference REACT
            rn_model = ResnetModule(arch_name=cfg.model.model_name,
                                    input_channels=cfg.model.input_channels,
                                    num_classes=10 if cfg.ind_dataset == "cifar10" else 43,
                                    spectral_norm=cfg.model.spectral_norm,
                                    dropblock=cfg.model.drop_block,
                                    dropblock_prob=cfg.model.dropblock_prob,
                                    dropblock_location = cfg.hook_location,
                                    dropblock_block_size=cfg.model.dropblock_block_size,
                                    dropout=cfg.model.dropout,
                                    dropout_prob=cfg.model.dropout_prob,
                                    loss_fn=cfg.model.loss_type,
                                    optimizer_lr=cfg.model.lr,
                                    optimizer_weight_decay=cfg.model.weight_decay,
                                    max_nro_epochs=cfg.trainer.epochs,
                                    activation=cfg.model.activation,
                                    avg_pool=cfg.model.avg_pool,
                                    ash=False,
                                    ash_percentile=cfg.ash_percentile,
                                    dice_precompute=False,
                                    dice_inference=False,
                                    dice_p=cfg.dice_p,
                                    dice_info=None,
                                    react_threshold=react_threshold,
                                    spectral_norm_only_fc=cfg.model.spectral_norm_only_fc,
                                    batch_norm=cfg.model.batch_norm
                                    )
            rn_model.load_from_checkpoint(checkpoint_path=cfg.model_path)
            rn_model.to(device)
            rn_model.eval()
            ind_valid_test_react = []
            for split, data_loader in ind_dataset_dict.items():
                if not split == "train":
                    print(f"\nReAct from InD {split}")
                    ind_valid_test_react.append(
                        get_energy_score(dnn_model=rn_model.model, input_dataloader=data_loader)
                    )
            ind_react_score = np.concatenate((ind_valid_test_react[0], ind_valid_test_react[1]))
            np.save(f"{save_dir}/{cfg.ind_dataset}_react", ind_react_score)

            for dataset_name, data_loaders in ood_datasets_dict.items():
                print(f"\nReAct from OoD {dataset_name}")
                ood_test_react_score = get_energy_score(dnn_model=rn_model.model, input_dataloader=data_loaders["test"])
                ood_valid_react_score = get_energy_score(dnn_model=rn_model.model,
                                                         input_dataloader=data_loaders["valid"])
                ood_react_score = np.concatenate((ood_test_react_score, ood_valid_react_score))
                np.save(f"{save_dir}/{dataset_name}_react", ood_react_score)
        ########
        # DICE
        if "dice" in cfg.baselines:
            # Inference DICE
            rn_model = ResnetModule(arch_name=cfg.model.model_name,
                                    input_channels=cfg.model.input_channels,
                                    num_classes=10 if cfg.ind_dataset == "cifar10" else 43,
                                    spectral_norm=cfg.model.spectral_norm,
                                    dropblock=cfg.model.drop_block,
                                    dropblock_prob=cfg.model.dropblock_prob,
                                    dropblock_location=cfg.hook_location,
                                    dropblock_block_size=cfg.model.dropblock_block_size,
                                    dropout=cfg.model.dropout,
                                    dropout_prob=cfg.model.dropout_prob,
                                    loss_fn=cfg.model.loss_type,
                                    optimizer_lr=cfg.model.lr,
                                    optimizer_weight_decay=cfg.model.weight_decay,
                                    max_nro_epochs=cfg.trainer.epochs,
                                    activation=cfg.model.activation,
                                    avg_pool=cfg.model.avg_pool,
                                    ash=False,
                                    ash_percentile=cfg.ash_percentile,
                                    dice_precompute=False,
                                    dice_inference=True,
                                    dice_p=cfg.dice_p,
                                    dice_info=dice_info_mean,
                                    react_threshold=None,
                                    spectral_norm_only_fc=cfg.model.spectral_norm_only_fc,
                                    batch_norm=cfg.model.batch_norm
                                    )
            rn_model.load_from_checkpoint(checkpoint_path=cfg.model_path)
            rn_model.to(device)
            rn_model.eval()
            ind_valid_test_dice = []
            for split, data_loader in ind_dataset_dict.items():
                if not split == "train":
                    print(f"\nDICE from InD {split}")
                    ind_valid_test_dice.append(
                        get_energy_score(dnn_model=rn_model.model, input_dataloader=data_loader)
                    )
            ind_dice_score = np.concatenate((ind_valid_test_dice[0], ind_valid_test_dice[1]))
            np.save(f"{save_dir}/{cfg.ind_dataset}_dice", ind_dice_score)

            for dataset_name, data_loaders in ood_datasets_dict.items():
                print(f"\nDICE from OoD {dataset_name}")
                ood_test_dice_score = get_energy_score(dnn_model=rn_model.model, input_dataloader=data_loaders["test"])
                ood_valid_dice_score = get_energy_score(dnn_model=rn_model.model, input_dataloader=data_loaders["valid"])
                ood_dice_score = np.concatenate((ood_test_dice_score, ood_valid_dice_score))
                np.save(f"{save_dir}/{dataset_name}_dice", ood_dice_score)
        #########
        # DICE + ReAct
        if "dice_react" in cfg.baselines:
            # Inference DICE
            rn_model = ResnetModule(arch_name=cfg.model.model_name,
                                    input_channels=cfg.model.input_channels,
                                    num_classes=10 if cfg.ind_dataset == "cifar10" else 43,
                                    spectral_norm=cfg.model.spectral_norm,
                                    dropblock=cfg.model.drop_block,
                                    dropblock_prob=cfg.model.dropblock_prob,
                                    dropblock_location=cfg.hook_location,
                                    dropblock_block_size=cfg.model.dropblock_block_size,
                                    dropout=cfg.model.dropout,
                                    dropout_prob=cfg.model.dropout_prob,
                                    loss_fn=cfg.model.loss_type,
                                    optimizer_lr=cfg.model.lr,
                                    optimizer_weight_decay=cfg.model.weight_decay,
                                    max_nro_epochs=cfg.trainer.epochs,
                                    activation=cfg.model.activation,
                                    avg_pool=cfg.model.avg_pool,
                                    ash=False,
                                    ash_percentile=cfg.ash_percentile,
                                    dice_precompute=False,
                                    dice_inference=True,
                                    dice_p=cfg.dice_p,
                                    dice_info=dice_info_mean,
                                    react_threshold=react_threshold,
                                    spectral_norm_only_fc=cfg.model.spectral_norm_only_fc,
                                    batch_norm=cfg.model.batch_norm
                                    )
            rn_model.load_from_checkpoint(checkpoint_path=cfg.model_path)
            rn_model.to(device)
            rn_model.eval()
            ind_valid_test_dice_react = []
            for split, data_loader in ind_dataset_dict.items():
                if not split == "train":
                    print(f"\nDICE + ReAct from InD {split}")
                    ind_valid_test_dice_react.append(
                        get_energy_score(dnn_model=rn_model.model, input_dataloader=data_loader)
                    )
            ind_dice_react_score = np.concatenate((ind_valid_test_dice_react[0], ind_valid_test_dice_react[1]))
            np.save(f"{save_dir}/{cfg.ind_dataset}_dice_react", ind_dice_react_score)

            for dataset_name, data_loaders in ood_datasets_dict.items():
                print(f"\nDICE + ReAct from OoD {dataset_name}")
                ood_test_dice_react_score = get_energy_score(dnn_model=rn_model.model, input_dataloader=data_loaders["test"])
                ood_valid_dice_react_score = get_energy_score(dnn_model=rn_model.model, input_dataloader=data_loaders["valid"])
                ood_dice_react_score = np.concatenate((ood_test_dice_react_score, ood_valid_dice_react_score))
                np.save(f"{save_dir}/{dataset_name}_dice_react", ood_dice_react_score)

    print("Done!")


if __name__ == '__main__':
    extract_and_save_mcd_samples()
