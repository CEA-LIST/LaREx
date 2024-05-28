import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import hydra
import mlflow
from omegaconf import DictConfig
from os.path import join as op_join
from tqdm import tqdm
from helper_functions import log_params_from_omegaconf_dict
from ls_ood_detect.uncertainty_estimation import get_predictive_uncertainty_score
from ls_ood_detect.metrics import get_hz_detector_results, \
    save_roc_ood_detector, save_scores_plots, get_pred_scores_plots, log_evaluate_lared_larem, \
    select_and_log_best_lared_larem
from ls_ood_detect.dimensionality_reduction import apply_pca_ds_split, apply_pca_transform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# If both next two flags are false, mlflow will create a local tracking uri for the experiment
# Upload analysis to the TDL server
UPLOAD_FROM_LOCAL_TO_SERVER = True
# Upload analysis ran on the TDL server
UPLOAD_FROM_SERVER_TO_SERVER = False
assert UPLOAD_FROM_SERVER_TO_SERVER + UPLOAD_FROM_LOCAL_TO_SERVER <= 1


@hydra.main(version_base=None, config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    # Get date-time to save df later
    current_date = cfg.log_dir.split("/")[-1]
    # Get samples folder
    mcd_samples_folder = f"./Mcd_samples/ind_{cfg.ind_dataset}/"
    save_dir = f"{mcd_samples_folder}{cfg.model_path.split('/')[2]}/{cfg.layer_type}"
    all_baselines = ["pred_h", "mi"]
    all_baselines.extend(cfg.baselines)
    ######################################################################
    # Load all data
    ######################################################################
    # Raw predictions
    ind_data_dict = {"valid_preds": torch.load(
        f=op_join(save_dir, f"{cfg.ind_dataset}_valid_mcd_preds.pt"),
        map_location=device
    ), "test_preds": torch.load(
        f=op_join(save_dir, f"{cfg.ind_dataset}_test_mcd_preds.pt"),
        map_location=device
    )}
    # Useful only to calculate the predictive entropy score and the mutual information
    ood_raw_preds_dict = {}
    for dataset_name in cfg.ood_datasets:
        if dataset_name == "anomalies":
            ood_raw_preds_dict[f"anomalies valid"] = torch.load(
                f=op_join(save_dir, f"{cfg.ind_dataset}_anomal_valid_mcd_preds.pt"), map_location=device
            )
            ood_raw_preds_dict[f"anomalies test"] = torch.load(
                f=op_join(save_dir, f"{cfg.ind_dataset}_anomal_test_mcd_preds.pt"), map_location=device
            )
        else:
            ood_raw_preds_dict[f"{dataset_name} valid"] = torch.load(
                f=op_join(save_dir, f"{dataset_name}_valid_mcd_preds.pt"), map_location=device
            )
            ood_raw_preds_dict[f"{dataset_name} test"] = torch.load(
                f=op_join(save_dir, f"{dataset_name}_test_mcd_preds.pt"), map_location=device
            )
    # Load all data baselines scores
    ood_scores_dict = {}
    for baseline in cfg.baselines:
        ind_data_dict[baseline] = np.load(file=op_join(save_dir, f"{cfg.ind_dataset}_{baseline}.npy"))
        for ood_dataset in cfg.ood_datasets:
            if ood_dataset == "anomalies":
                ood_scores_dict[f"{ood_dataset} {baseline}"] = np.load(
                    file=op_join(save_dir, f"{cfg.ind_dataset}_anomal_{baseline}.npy")
                )
            else:
                ood_scores_dict[f"{ood_dataset} {baseline}"] = np.load(
                    file=op_join(save_dir, f"{ood_dataset}_{baseline}.npy")
                )
    # Load InD entropies
    ind_data_dict["h_z_train"] = np.load(file=op_join(save_dir, f"{cfg.ind_dataset}_h_z_train.npy"))
    ind_data_dict["h_z"] = np.load(file=op_join(save_dir, f"{cfg.ind_dataset}_h_z.npy"))

    # OoD Entropies
    ood_entropies_dict = {}
    for ood_dataset in cfg.ood_datasets:
        if ood_dataset == "anomalies":
            ood_entropies_dict[ood_dataset] = np.load(file=op_join(save_dir, f"{cfg.ind_dataset}_anomal_h_z.npy"))
        else:
            ood_entropies_dict[ood_dataset] = np.load(file=op_join(save_dir, f"{ood_dataset}_h_z.npy"))
    # assert False
    # start_n_h_comps = 20
    # n_h_comps = 15
    # violin_plot_x = np.arange(n_h_comps)
    # fig = plt.figure(figsize=(8, 6))
    # plt.violinplot(dataset=ind_data_dict["h_z_train"][:, start_n_h_comps:start_n_h_comps+n_h_comps], positions=violin_plot_x)
    # plt.violinplot(dataset=ood_entropies_dict["textures"][:, start_n_h_comps:start_n_h_comps+n_h_comps], positions=violin_plot_x)
    # plt.legend()
    #######################################################################
    # Setup MLFLow
    #######################################################################
    # Setup MLFlow for experiment tracking
    # MlFlow configuration
    experiment_name = cfg.logger.mlflow.experiment_name
    if UPLOAD_FROM_LOCAL_TO_SERVER:
        mlflow.set_tracking_uri("http://10.8.33.50:5050")
    elif UPLOAD_FROM_SERVER_TO_SERVER:
        mlflow.set_tracking_uri("http://127.0.0.1:5051")
    existing_exp = mlflow.get_experiment_by_name(experiment_name)
    if not existing_exp:
        mlflow.create_experiment(
            name=experiment_name,
        )
    experiment = mlflow.set_experiment(experiment_name=experiment_name)

    ############################################################################################################
    ############################################################################################################
    ##########################################################################
    # Start the evaluation run
    ##########################################################################
    # Define mlflow run to log metrics and parameters
    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        print("run_id: {}; status: {}".format(run.info.run_id, run.info.status))
        # Log parameters with mlflow
        log_params_from_omegaconf_dict(cfg)
        ##########################################################################################
        ########################################################
        # Evaluate baselines
        ########################################################
        ########################
        # Predictive uncertainty - mutual information
        # InD set
        ind_data_dict["pred_h"], ind_data_dict["mi"] = get_predictive_uncertainty_score(
            input_samples=torch.cat((ind_data_dict["valid_preds"], ind_data_dict["test_preds"]), dim=0),
            mcd_nro_samples=cfg.mcd_n_samples
        )
        ind_data_dict["pred_h"], ind_data_dict["mi"] = \
            ind_data_dict["pred_h"].cpu().numpy(), ind_data_dict["mi"].cpu().numpy()
        # OoD datasets
        for ood_dataset in cfg.ood_datasets:
            ood_scores_dict[f"{ood_dataset} pred_h"], ood_scores_dict[f"{ood_dataset} mi"] = \
                get_predictive_uncertainty_score(
                    input_samples=torch.cat((ood_raw_preds_dict[f"{ood_dataset} valid"],
                                             ood_raw_preds_dict[f"{ood_dataset} test"]), dim=0),
                    mcd_nro_samples=cfg.mcd_n_samples
                )
            ood_scores_dict[f"{ood_dataset} pred_h"], ood_scores_dict[f"{ood_dataset} mi"] = \
                ood_scores_dict[f"{ood_dataset} pred_h"].cpu().numpy(), ood_scores_dict[
                    f"{ood_dataset} mi"].cpu().numpy()

        # Dictionary that defines experiments names, InD and OoD datasets
        # We use some negative uncertainty scores to align with the convention that positive
        # (in-distribution) samples have higher scores (see plots)
        baselines_experiments = {}
        for baseline in all_baselines:
            for ood_dataset in cfg.ood_datasets:
                if baseline == "pred_h" or baseline == "mi":
                    baselines_experiments[f"{ood_dataset} {baseline}"] = {
                        "InD": -ind_data_dict[baseline],
                        "OoD": -ood_scores_dict[f"{ood_dataset} {baseline}"]
                    }
                else:
                    baselines_experiments[f"{ood_dataset} {baseline}"] = {
                        "InD": ind_data_dict[baseline],
                        "OoD": ood_scores_dict[f"{ood_dataset} {baseline}"]
                    }

        baselines_plots = {}
        for baseline in all_baselines:
            baselines_plots[baseline_name_dict[baseline]["plot_title"]] = {"InD": ind_data_dict[baseline]}
            baselines_plots[baseline_name_dict[baseline]["plot_title"]]["x_axis"] = \
                baseline_name_dict[baseline]["x_axis"]
            baselines_plots[baseline_name_dict[baseline]["plot_title"]]["plot_name"] = \
                baseline_name_dict[baseline]["plot_name"]
            for ood_dataset in cfg.ood_datasets:
                baselines_plots[baseline_name_dict[baseline]["plot_title"]][ood_dataset] = \
                    ood_scores_dict[f"{ood_dataset} {baseline}"]

        # Make all baselines plots
        for plot_title, experiment in tqdm(baselines_plots.items(), desc="Plotting baselines"):
            # Plot score values predictive entropy
            pred_score_plot = get_pred_scores_plots(experiment,
                                                    cfg.ood_datasets,
                                                    title=plot_title,
                                                    ind_dataset_name=cfg.ind_dataset)
            mlflow.log_figure(figure=pred_score_plot.figure,
                              artifact_file=f"figs/{experiment['plot_name']}.png")

        # Initialize df to store all the results
        overall_metrics_df = pd.DataFrame(columns=['auroc', 'fpr@95', 'aupr',
                                                   'fpr', 'tpr', 'roc_thresholds',
                                                   'precision', 'recall', 'pr_thresholds'])
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

        # Clean memory
        del baselines_plots
        del baselines_experiments
        del ood_scores_dict
        del ood_raw_preds_dict

        ######################################################
        # Evaluate OoD detection method LaRED & LaREM
        ######################################################
        print("LaRED & LaREM running...")
        # Perform evaluation with the complete vector of latent representations
        r_df, ind_lared_scores, ood_lared_scores_dict = log_evaluate_lared_larem(
            ind_train_h_z=ind_data_dict["h_z_train"],
            ind_test_h_z=ind_data_dict["h_z"],
            ood_h_z_dict=ood_entropies_dict,
            experiment_name_extension="",
            return_density_scores=True,
            mlflow_logging=True
        )
        # Add results to df
        overall_metrics_df = overall_metrics_df.append(r_df)
        # Plots comparison of densities
        lared_scores_plots_dict = save_scores_plots(ind_lared_scores,
                                                    ood_lared_scores_dict,
                                                    cfg.ood_datasets,
                                                    cfg.ind_dataset)
        for plot_name, plot in lared_scores_plots_dict.items():
            mlflow.log_figure(figure=plot.figure,
                              artifact_file=f"figs/{plot_name}.png")

        # Perform evaluation with PCA reduced vectors
        for n_components in tqdm(cfg.n_pca_components, desc="Evaluating PCA"):
            # Perform PCA dimension reduction
            pca_h_z_ind_train, pca_transformation = apply_pca_ds_split(
                samples=ind_data_dict["h_z_train"],
                nro_components=n_components
            )
            pca_h_z_ind_test = apply_pca_transform(ind_data_dict["h_z"], pca_transformation)
            ood_pca_dict = {}
            for ood_dataset in cfg.ood_datasets:
                ood_pca_dict[ood_dataset] = apply_pca_transform(ood_entropies_dict[ood_dataset], pca_transformation)

            r_df = log_evaluate_lared_larem(
                ind_train_h_z=pca_h_z_ind_train,
                ind_test_h_z=pca_h_z_ind_test,
                ood_h_z_dict=ood_pca_dict,
                experiment_name_extension=f" PCA {n_components}",
                return_density_scores=False,
                log_step=n_components,
                mlflow_logging=True
            )
            # Add results to df
            overall_metrics_df = overall_metrics_df.append(r_df)

        overall_metrics_df_name = f"./results_csvs/{current_date}_experiment.csv.gz"
        overall_metrics_df.to_csv(path_or_buf=overall_metrics_df_name, compression="gzip")
        mlflow.log_artifact(overall_metrics_df_name)
        if cfg.layer_type == "Conv":
            hook_layer_type = "DropBlock"
        else:
            hook_layer_type = "Dropout"

        # Plot Roc curves together, by OoD dataset
        for ood_dataset in cfg.ood_datasets:
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
                if ood_dataset in row_name and "PCA" not in row_name:
                    temp_df = temp_df.append(overall_metrics_df.loc[row_name])
                    temp_df.rename(index={row_name: row_name.split(ood_dataset)[1]}, inplace=True)
                elif ood_dataset in row_name and "PCA" in row_name and "LaREM" in row_name:
                    temp_df_pca_larem = temp_df_pca_larem.append(overall_metrics_df.loc[row_name])
                    temp_df_pca_larem.rename(index={row_name: row_name.split(ood_dataset)[1]}, inplace=True)
                elif ood_dataset in row_name and "PCA" in row_name and "LaRED" in row_name:
                    temp_df_pca_lared = temp_df_pca_lared.append(overall_metrics_df.loc[row_name])
                    temp_df_pca_lared.rename(index={row_name: row_name.split(ood_dataset)[1]}, inplace=True)
            # Plot ROC curve
            roc_curve = save_roc_ood_detector(
                results_table=temp_df,
                plot_title=f"ROC {cfg.ind_dataset} vs {ood_dataset} {hook_layer_type} layer"
            )
            # Log the plot with mlflow
            mlflow.log_figure(figure=roc_curve,
                              artifact_file=f"figs/roc_{ood_dataset}.png")
            roc_curve_pca_larem = save_roc_ood_detector(
                results_table=temp_df_pca_larem,
                plot_title=f"ROC {cfg.ind_dataset} vs {ood_dataset} LareM PCA {hook_layer_type} layer"
            )
            # Log the plot with mlflow
            mlflow.log_figure(figure=roc_curve_pca_larem,
                              artifact_file=f"figs/roc_{ood_dataset}_pca_larem.png")
            roc_curve_pca_lared = save_roc_ood_detector(
                results_table=temp_df_pca_lared,
                plot_title=f"ROC {cfg.ind_dataset} vs {ood_dataset} LareD PCA {cfg.layer_type} layer"
            )
            # Log the plot with mlflow
            mlflow.log_figure(figure=roc_curve_pca_lared,
                              artifact_file=f"figs/roc_{ood_dataset}_pca_lared.png")

        # We collect all metrics to estimate global performance of all metrics
        all_aurocs = []
        all_auprs = []
        all_fprs = []
        # Extract mean for each baseline across datasets
        for baseline in all_baselines:
            temp_df = pd.DataFrame(columns=['auroc', 'fpr@95', 'aupr',
                                            'fpr', 'tpr', 'roc_thresholds',
                                            'precision', 'recall', 'pr_thresholds'])
            for row_name in overall_metrics_df.index:
                if baseline in row_name and "anomalies" not in row_name:
                    temp_df = temp_df.append(overall_metrics_df.loc[row_name])
                    temp_df.rename(index={row_name: row_name.split(baseline)[0]}, inplace=True)

            mlflow.log_metric(f"{baseline}_auroc_mean", temp_df["auroc"].mean())
            all_aurocs.append(temp_df["auroc"].mean())
            mlflow.log_metric(f"{baseline}_auroc_std", temp_df["auroc"].std())
            mlflow.log_metric(f"{baseline}_aupr_mean", temp_df["aupr"].mean())
            all_auprs.append(temp_df["aupr"].mean())
            mlflow.log_metric(f"{baseline}_aupr_std", temp_df["aupr"].std())
            mlflow.log_metric(f"{baseline}_fpr95_mean", temp_df["fpr@95"].mean())
            all_fprs.append(temp_df["fpr@95"].mean())
            mlflow.log_metric(f"{baseline}_fpr95_std", temp_df["fpr@95"].std())

        # Extract mean for LaRED & LaREM across datasets
        # LaRED
        auroc_lared, aupr_lared, fpr_lared = select_and_log_best_lared_larem(
            overall_metrics_df, cfg.n_pca_components, technique="LaRED", log_mlflow=True
        )
        all_aurocs.append(auroc_lared)
        all_auprs.append(aupr_lared)
        all_fprs.append(fpr_lared)
        # LaREM
        auroc_larem, aupr_larem, fpr_larem = select_and_log_best_lared_larem(
            overall_metrics_df, cfg.n_pca_components, technique="LaREM", log_mlflow=True
        )
        all_aurocs.append(auroc_larem)
        all_auprs.append(aupr_larem)
        all_fprs.append(fpr_larem)
        mlflow.log_metric(f"global_auroc_mean", np.mean(all_aurocs))
        mlflow.log_metric(f"global_auroc_std", np.std(all_aurocs))
        mlflow.log_metric(f"global_aupr_mean", np.mean(all_auprs))
        mlflow.log_metric(f"global_aupr_std", np.std(all_auprs))
        mlflow.log_metric(f"global_fpr_mean", np.mean(all_fprs))
        mlflow.log_metric(f"global_fpr_std", np.std(all_fprs))

        mlflow.end_run()


baseline_name_dict = {
    "pred_h": {
        "plot_title": "Predictive H distribution",
        "x_axis": "Predictive H score",
        "plot_name": "pred_h"
    },
    "mi": {
        "plot_title": "Predictive MI distribution",
        "x_axis": "Predictive MI score",
        "plot_name": "pred_mi"
    },
    "msp": {
        "plot_title": "Predictive MSP distribution",
        "x_axis": "Predictive MSP score",
        "plot_name": "pred_msp"
    },
    "energy": {
        "plot_title": "Predictive energy score distribution",
        "x_axis": "Predictive energy score",
        "plot_name": "pred_energy"
    },
    "mdist": {
        "plot_title": "Mahalanobis Distance distribution",
        "x_axis": "Mahalanobis Distance score",
        "plot_name": "pred_mdist"
    },
    "knn": {
        "plot_title": "kNN distance distribution",
        "x_axis": "kNN Distance score",
        "plot_name": "pred_knn"
    },
    "ash": {
        "plot_title": "ASH score distribution",
        "x_axis": "ASH score",
        "plot_name": "ash_score"
    },
    "dice": {
        "plot_title": "DICE score distribution",
        "x_axis": "DICE score",
        "plot_name": "dice_score"
    },
    "react": {
        "plot_title": "ReAct score distribution",
        "x_axis": "ReAct score",
        "plot_name": "react_score"
    },
    "dice_react": {
        "plot_title": "DICE + ReAct score distribution",
        "x_axis": "DICE + ReAct score",
        "plot_name": "dice_react_score"
    }
}

if __name__ == '__main__':
    main()
