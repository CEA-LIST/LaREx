import torch
import numpy as np
from ls_ood_detect_cea.rcnn import get_energy_score_rcnn
from torch.utils.data import DataLoader


def save_energy_scores_baselines(predictor: torch.nn.Module,
                                 data_loader: DataLoader,
                                 baseline_name: str,
                                 save_foldar_name: str,
                                 ds_name: str,
                                 ds_type: str
                                 ):
    assert ds_type in ("ind", "ood")
    print(f"\n{baseline_name} from {ds_type} {ds_name}")
    raw_test_energy, filtered_test_energy = \
        get_energy_score_rcnn(dnn_model=predictor, input_dataloader=data_loader)
    np.save(f"./{save_foldar_name}/{ds_name}_{ds_type}_raw_{baseline_name}", raw_test_energy)
    np.save(f"./{save_foldar_name}/{ds_name}_{ds_type}_filtered_{baseline_name}", filtered_test_energy)

