import mlflow
import numpy as np
import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from os.path import join as op_join
from pytorch_lightning import seed_everything as pl_seed_everything
from ls_ood_detect.uncertainty_estimation import get_dl_h_z
from ls_ood_detect.dimensionality_reduction import plot_samples_pacmap, apply_pca_ds_split, apply_pca_transform
from ls_ood_detect.metrics import get_hz_detector_results, save_roc_ood_detector, get_pred_scores_plots, log_evaluate_lared_larem, baseline_name_dict
from tqdm import tqdm
from param_logging import log_params_from_omegaconf_dict
from mcd_helper_fns import reduce_mcd_samples
import warnings

# Filter the append warning from pandas
warnings.simplefilter(action='ignore', category=FutureWarning)

# If both next two flags are false, mlflow will create a local tracking uri for the experiment
# Upload analysis to the TDL server
UPLOAD_FROM_LOCAL_TO_SERVER = True
# Upload analysis ran on the TDL server
UPLOAD_FROM_SERVER_TO_SERVER = False
assert UPLOAD_FROM_SERVER_TO_SERVER + UPLOAD_FROM_LOCAL_TO_SERVER <= 1
# Perform analysis either on RCNN or RESNET
RCNN = True
RESNET = False
assert RCNN + RESNET == 1
if RCNN:
    config_file = "config_rcnn.yaml"
else:
    config_file = "config.yaml"
BASELINES = ["pred_h", "mi", "ash", "react", "dice", "dice_react", "msp", "energy"]
# energy_based_baselines = ["ash", "react", "dice", "dice_react", "energy"]
full_names_baselines = ["pred_h", "mi", "msp", "filtered_energy", "filtered_ash",
                        "filtered_react", "filtered_dice", "filtered_dice_react",
                        "raw_energy", "raw_ash", "raw_react", "raw_dice", "raw_dice_react"]


@hydra.main(version_base=None, config_path="configs/MCD_evaluation", config_name=config_file)
def main(cfg: DictConfig) -> None:
    """
    This function performs analysis on already calculated MCD samples in another script.
    Evaluates BDD as In distribution dataset against either COCO or Openimages as described in the VOS repository.
    This script assumes without checking that the number of MCD runs is exactly the same for the
    InD (BDD) data and the OoD data.
    :return: None
    """
    ############################
    #      Seed Everything     #
    ############################
    pl_seed_everything(cfg.seed)
    # Select device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ###############################################
    # Load precalculated MCD samples              #
    ###############################################
    # Inspect correct naming of files and dataset
    assert cfg.ood_dataset in cfg.ood_mcd_samples and cfg.ood_dataset in cfg.ood_entropy_test, "OoO Dataset name and preloaded files must coincide"
    assert cfg.layer_type in cfg.ood_mcd_samples and cfg.layer_type in cfg.ood_entropy_test, "Location of samples must coincide with filename"
    assert "h_z" in cfg.ood_entropy_test and "h_z" not in cfg.ood_mcd_samples
    assert cfg.ood_dataset in ('coco', 'openimages', "gtsrb", "svhn", "cifar10", "bdd", "voc")
    ind_valid_mc_samples = torch.load(f=op_join(cfg.data_dir, cfg.bdd_valid_mcd_samples),
                                      map_location=device)
    ind_test_mc_samples = torch.load(f=op_join(cfg.data_dir, cfg.bdd_test_mcd_samples),
                                     map_location=device)
    ood_test_mc_samples = torch.load(f=op_join(cfg.data_dir, cfg.ood_mcd_samples),
                                     map_location=device)

    ##############################################################
    # Select number of MCD runs to use                           #
    ##############################################################
    assert cfg.n_mcd_runs <= cfg.precomputed_mcd_runs, "n_mcd_runs must be less than or equal to the precomputed runs"
    if cfg.n_mcd_runs < cfg.precomputed_mcd_runs:
        ind_valid_mc_samples, ind_test_mc_samples, ood_test_mc_samples = reduce_mcd_samples(
            bdd_valid_mc_samples=ind_valid_mc_samples,
            bdd_test_mc_samples=ind_test_mc_samples,
            ood_test_mc_samples=ood_test_mc_samples,
            precomputed_mcd_runs=cfg.precomputed_mcd_runs,
            n_mcd_runs=cfg.n_mcd_runs
        )

    #################################################################
    # Select number of proposals from RPN to use
    #################################################################
    max_n_proposals = cfg.max_n_proposals
    assert cfg.use_n_proposals <= max_n_proposals, "use_n_proposals must be less than or equal to the max proposals"
    # Compute entropy only of n_mcd_runs < cfg.precomputed_mcd_runs, otherwise, load precomputed entropy
    if cfg.use_n_proposals < max_n_proposals and cfg.n_mcd_runs < cfg.precomputed_mcd_runs:
        # Randomly select the columns to keep:
        columns_to_keep = torch.randperm(max_n_proposals)[:cfg.use_n_proposals]
        ind_valid_mc_samples = ind_valid_mc_samples[:, columns_to_keep]
        ind_test_mc_samples = ind_test_mc_samples[:, columns_to_keep]
        ood_test_mc_samples = ood_test_mc_samples[:, columns_to_keep]

    #####################################################################
    # Compute entropy
    #####################################################################
    if cfg.n_mcd_runs == cfg.precomputed_mcd_runs:
        # Load precomputed entropy in this case
        bdd_valid_h_z_np = np.load(file=op_join(cfg.data_dir, cfg.bdd_entropy_valid))
        bdd_test_h_z_np = np.load(file=op_join(cfg.data_dir, cfg.bdd_entropy_test))
        ood_h_z_np = np.load(file=op_join(cfg.data_dir, cfg.ood_entropy_test))
        # Reduce number of proposals here, to avoid recalculations
        if cfg.use_n_proposals < max_n_proposals:
            columns_to_keep = torch.randperm(max_n_proposals)[:cfg.use_n_proposals]
            bdd_valid_h_z_np = bdd_valid_h_z_np[:, columns_to_keep]
            bdd_test_h_z_np = bdd_test_h_z_np[:, columns_to_keep]
            ood_h_z_np = ood_h_z_np[:, columns_to_keep]
    # Calculate entropy only if cfg.n_mcd_runs < cfg.precomputed_mcd_runs
    else:
        # Calculate entropy for bdd valid set
        _, bdd_valid_h_z_np = get_dl_h_z(ind_valid_mc_samples,
                                         mcd_samples_nro=cfg.n_mcd_runs)
        # Calculate entropy bdd test set
        _, bdd_test_h_z_np = get_dl_h_z(ind_test_mc_samples,
                                        mcd_samples_nro=cfg.n_mcd_runs)
        # Calculate entropy ood test set
        _, ood_h_z_np = get_dl_h_z(ood_test_mc_samples,
                                   mcd_samples_nro=cfg.n_mcd_runs)

    # Since these data is no longer needed we can delete it
    del ind_valid_mc_samples
    del ind_test_mc_samples
    del ood_test_mc_samples

    ######################################################################
    # Load baselines data
    ######################################################################
    ind_data_dict = dict()
    ood_scores_dict = {}
    for baseline in full_names_baselines:
        ind_data_dict[baseline] = np.load(
            file=op_join(cfg.data_dir, f"{cfg.layer_type}/{cfg.ind_dataset}_ind_{baseline}.npy"))
        ood_scores_dict[f"{cfg.ood_dataset} {baseline}"] = np.load(
            file=op_join(cfg.data_dir, f"{cfg.layer_type}/{cfg.ood_dataset}_ood_{baseline}.npy")
        )

    #######################################################################
    # Setup MLFLow
    #######################################################################
    # Setup MLFlow for experiment tracking
    # MlFlow configuration
    experiment_name = cfg.logger.mlflow.experiment_name
    if UPLOAD_FROM_LOCAL_TO_SERVER:
        mlflow.set_tracking_uri("http://XXXXXXXXXXX")
    elif UPLOAD_FROM_SERVER_TO_SERVER:
        mlflow.set_tracking_uri("http://XXXXXXXXXXXXX")
    existing_exp = mlflow.get_experiment_by_name(experiment_name)
    if not existing_exp:
        mlflow.create_experiment(
            name=experiment_name,
        )
    experiment = mlflow.set_experiment(experiment_name=experiment_name)
    # mlflow.set_tracking_uri(cfg.logger.mlflow.tracking_uri)
    # Let us define a run name automatically.
    if cfg.ood_dataset == "coco":
        mlflow_run_dataset = "cc"
    elif cfg.ood_dataset == "openimages":
        mlflow_run_dataset = "oi"
    elif cfg.ood_dataset == "gtsrb":
        mlflow_run_dataset = "gtsrb"
    elif cfg.ood_dataset == "svhn":
        mlflow_run_dataset = "svhn"
    elif cfg.ood_dataset == "bdd":
        mlflow_run_dataset = "bdd"
    elif cfg.ood_dataset == "voc":
        mlflow_run_dataset = "voc"
    else:
        mlflow_run_dataset = "cifar10"
    mlflow_run_name = f"{mlflow_run_dataset}_{cfg.layer_type}_{cfg.n_mcd_runs}_mcd_{cfg.use_n_proposals}"

    ##########################################################################
    # Start the evaluation run
    ##########################################################################
    # Define mlflow run to log metrics and parameters
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=mlflow_run_name) as run:
        # Log parameters with mlflow
        log_params_from_omegaconf_dict(cfg)
        # Check entropy 2D projection
        pacmap_2d_proj_plot = plot_samples_pacmap(samples_ind=bdd_test_h_z_np,
                                                  samples_ood=ood_h_z_np,
                                                  neighbors=cfg.n_pacmap_neighbors,
                                                  title=cfg.ind_dataset + " - " + cfg.ood_dataset + " : $\hat{H}_{\phi}(z_i \mid x)$",
                                                  return_figure=True)
        mlflow.log_figure(figure=pacmap_2d_proj_plot,
                          artifact_file="figs/h_z_pacmap.png")
        #######################################################################
        # Start PCA LaRED & LaREM evaluation
        #######################################################################
        overall_metrics_df = pd.DataFrame(columns=['auroc', 'fpr@95', 'aupr',
                                                   'fpr', 'tpr', 'roc_thresholds',
                                                   'precision', 'recall', 'pr_thresholds'])
        for n_components in tqdm(cfg.n_pca_components, desc="Evaluating PCA"):
            # Perform PCA dimension reduction
            pca_h_z_ind_valid_samples, pca_transformation = apply_pca_ds_split(samples=bdd_valid_h_z_np,
                                                                               nro_components=n_components)
            pca_h_z_ind_test_samples = apply_pca_transform(bdd_test_h_z_np, pca_transformation)
            ood_pca_dict = {
                cfg.ood_dataset: apply_pca_transform(ood_h_z_np, pca_transformation)
            }
            r_df = log_evaluate_lared_larem(
                ind_train_h_z=pca_h_z_ind_valid_samples,
                ind_test_h_z=pca_h_z_ind_test_samples,
                ood_h_z_dict=ood_pca_dict,
                experiment_name_extension=f" PCA {n_components}",
                return_density_scores=False,
                log_step=n_components
            )
            # Add results to df
            overall_metrics_df = overall_metrics_df.append(r_df)

        ##################################################
        # Baselines evaluation
        ##################################################
        baselines_experiments = {}
        for baseline in full_names_baselines:
            if baseline == "pred_h" or baseline == "mi":
                baselines_experiments[f"{cfg.ood_dataset} {baseline}"] = {
                    "InD": -ind_data_dict[baseline],
                    "OoD": -ood_scores_dict[f"{cfg.ood_dataset} {baseline}"]
                }
            else:
                baselines_experiments[f"{cfg.ood_dataset} {baseline}"] = {
                    "InD": ind_data_dict[baseline],
                    "OoD": ood_scores_dict[f"{cfg.ood_dataset} {baseline}"]
                }

        baselines_plots = {}
        for baseline in full_names_baselines:
            baselines_plots[baseline_name_dict[baseline]["plot_title"]] = {"InD": ind_data_dict[baseline]}
            baselines_plots[baseline_name_dict[baseline]["plot_title"]]["x_axis"] = \
                baseline_name_dict[baseline]["x_axis"]
            baselines_plots[baseline_name_dict[baseline]["plot_title"]]["plot_name"] = \
                baseline_name_dict[baseline]["plot_name"]
            baselines_plots[baseline_name_dict[baseline]["plot_title"]][cfg.ood_dataset] = \
                ood_scores_dict[f"{cfg.ood_dataset} {baseline}"]

        # Make all baselines plots
        for plot_title, experiment in tqdm(baselines_plots.items(), desc="Plotting baselines"):
            # Plot score values predictive entropy
            pred_score_plot = get_pred_scores_plots(experiment,
                                                    [cfg.ood_dataset],
                                                    title=plot_title,
                                                    ind_dataset_name=cfg.ind_dataset)
            mlflow.log_figure(figure=pred_score_plot.figure,
                              artifact_file=f"figs/{experiment['plot_name']}.png")

        # Log all baselines experiments
        for experiment_name, experiment in tqdm(baselines_experiments.items(), desc="Logging baselines"):
            r_df, r_mlflow = get_hz_detector_results(detect_exp_name=experiment_name,
                                                     ind_samples_scores=experiment["InD"],
                                                     ood_samples_scores=experiment["OoD"],
                                                     return_results_for_mlflow=True)
            r_mlflow = dict([(f"{experiment_name}_{k}", v) for k, v in r_mlflow.items()])
            mlflow.log_metrics(r_mlflow)
            # Plot each ROC curve individually LEAVE COMMENTED
            # roc_curve = save_roc_ood_detector(
            #     results_table=r_df,
            #     plot_title=f"ROC {cfg.ind_dataset} vs {experiment_name} {cfg.layer_type} layer"
            # )
            # mlflow.log_figure(figure=roc_curve,
            #                   artifact_file=f"figs/roc_{experiment_name}.png")
            # END COMMENTED SECTION
            overall_metrics_df = overall_metrics_df.append(r_df)

        ############################################################
        # Plotting baselines and LaRED, LaREM
        ############################################################
        temp_df = pd.DataFrame(columns=['auroc', 'fpr@95', 'aupr',
                                        'fpr', 'tpr', 'roc_thresholds',
                                        'precision', 'recall', 'pr_thresholds'])
        temp_df_pca_lared = pd.DataFrame(columns=['auroc', 'fpr@95', 'aupr',
                                                  'fpr', 'tpr', 'roc_thresholds',
                                                  'precision', 'recall', 'pr_thresholds'])
        temp_df_pca_larem = pd.DataFrame(columns=['auroc', 'fpr@95', 'aupr',
                                                  'fpr', 'tpr', 'roc_thresholds',
                                                  'precision', 'recall', 'pr_thresholds'])
        for row_name in overall_metrics_df.index:
            if cfg.ood_dataset in row_name and "PCA" not in row_name:
                temp_df = temp_df.append(overall_metrics_df.loc[row_name])
                temp_df.rename(index={row_name: row_name.split(cfg.ood_dataset)[1]}, inplace=True)
            elif cfg.ood_dataset in row_name and "PCA" in row_name and "LaREM" in row_name:
                temp_df_pca_larem = temp_df_pca_larem.append(overall_metrics_df.loc[row_name])
                temp_df_pca_larem.rename(index={row_name: row_name.split(cfg.ood_dataset)[1]}, inplace=True)
            elif cfg.ood_dataset in row_name and "PCA" in row_name and "LaRED" in row_name:
                temp_df_pca_lared = temp_df_pca_lared.append(overall_metrics_df.loc[row_name])
                temp_df_pca_lared.rename(index={row_name: row_name.split(cfg.ood_dataset)[1]}, inplace=True)

        # Choose best LaRED and LaREM to plot along baselines
        best_lared_index = temp_df_pca_lared[temp_df_pca_lared.auroc == temp_df_pca_lared.auroc.max()].index[0]
        best_larem_index = temp_df_pca_larem[temp_df_pca_larem.auroc == temp_df_pca_larem.auroc.max()].index[0]
        # Add these to the plots temp df
        temp_df = temp_df.append(overall_metrics_df.loc[f"{cfg.ood_dataset}{best_lared_index}"])
        temp_df = temp_df.append(overall_metrics_df.loc[f"{cfg.ood_dataset}{best_larem_index}"])
        # Plot ROC curve
        roc_curve = save_roc_ood_detector(
            results_table=temp_df,
            plot_title=f"ROC {cfg.ind_dataset} vs {cfg.ood_dataset} {cfg.layer_type} layer"
        )
        # Log the plot with mlflow
        mlflow.log_figure(figure=roc_curve,
                          artifact_file=f"figs/roc_{cfg.ood_dataset}.png")
        roc_curve_pca_larem = save_roc_ood_detector(
            results_table=temp_df_pca_larem,
            plot_title=f"ROC {cfg.ind_dataset} vs {cfg.ood_dataset} LareM PCA {cfg.layer_type} layer"
        )
        # Log the plot with mlflow
        mlflow.log_figure(figure=roc_curve_pca_larem,
                          artifact_file=f"figs/roc_{cfg.ood_dataset}_pca_larem.png")
        roc_curve_pca_lared = save_roc_ood_detector(
            results_table=temp_df_pca_lared,
            plot_title=f"ROC {cfg.ind_dataset} vs {cfg.ood_dataset} LareD PCA {cfg.layer_type} layer"
        )
        # Log the plot with mlflow
        mlflow.log_figure(figure=roc_curve_pca_lared,
                          artifact_file=f"figs/roc_{cfg.ood_dataset}_pca_lared.png")

        mlflow.end_run()


if __name__ == "__main__":
    main()
