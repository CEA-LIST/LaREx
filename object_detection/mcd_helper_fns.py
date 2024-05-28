from typing import Tuple
import torch

# Ugly but fast way to test to hook the backbone: get raw output, apply dropblock
# inside the following function
dropout_ext = torch.nn.Dropout(p=0.5)


def adjust_mlflow_results_name(data_dict: dict, technique_name: str) -> dict:
    """
    This function simply adds the name of the dimension reduciton technique at the end of the metrics names,
    In order to facilitate analysis with mlflow
    :param data_dict: Metrics dictionary
    :param technique_name: Either pca or pm (pacmap)
    :return: Dictionary with changed keys
    """
    new_dict = dict()
    for k, v in data_dict.items():
        new_dict[k + f'_{technique_name}'] = v
    return new_dict


def reduce_mcd_samples(bdd_valid_mc_samples: torch.Tensor,
                       bdd_test_mc_samples: torch.Tensor,
                       ood_test_mc_samples: torch.Tensor,
                       precomputed_mcd_runs: int,
                       n_mcd_runs: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    This function takes precomputed Monte Carlo Dropout samples, and returns a smaller number of samples to work with
    :param bdd_valid_mc_samples:
    :param bdd_test_mc_samples:
    :param ood_test_mc_samples:
    :param precomputed_mcd_runs:
    :param n_mcd_runs:
    :return: Ind valid and test sets, OoD set
    """
    n_samples_bdd_valid = int(bdd_valid_mc_samples.shape[0] / precomputed_mcd_runs)
    n_samples_bdd_test = int(bdd_test_mc_samples.shape[0] / precomputed_mcd_runs)
    n_samples_ood = int(ood_test_mc_samples.shape[0] / precomputed_mcd_runs)
    # Reshape to easily subset mcd samples
    reshaped_bdd_valid = bdd_valid_mc_samples.reshape(n_samples_bdd_valid,
                                                      precomputed_mcd_runs,
                                                      bdd_valid_mc_samples.shape[1])
    reshaped_bdd_test = bdd_test_mc_samples.reshape(n_samples_bdd_test,
                                                    precomputed_mcd_runs,
                                                    bdd_test_mc_samples.shape[1])
    reshaped_ood_test = ood_test_mc_samples.reshape(n_samples_ood,
                                                    precomputed_mcd_runs,
                                                    ood_test_mc_samples.shape[1])
    # Select the desired number of samples to take
    bdd_valid_mc_samples = reshaped_bdd_valid[:, :n_mcd_runs, :].reshape(n_samples_bdd_valid * n_mcd_runs,
                                                                         bdd_valid_mc_samples.shape[1])
    bdd_test_mc_samples = reshaped_bdd_test[:, :n_mcd_runs, :].reshape(n_samples_bdd_test * n_mcd_runs,
                                                                       bdd_test_mc_samples.shape[1])
    ood_test_mc_samples = reshaped_ood_test[:, :n_mcd_runs, :].reshape(n_samples_ood * n_mcd_runs,
                                                                       ood_test_mc_samples.shape[1])
    return bdd_valid_mc_samples, bdd_test_mc_samples, ood_test_mc_samples
