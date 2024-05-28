from typing import Union, List, Any, Callable, Dict, Optional, Tuple
import numpy as np
import torch
import torch.utils.data as torchdata
from detectron2.data import get_detection_dataset_dicts
from detectron2.data import DatasetMapper, DatasetFromList, MapDataset
from detectron2.data.build import trivial_batch_collator
from detectron2.data.samplers import InferenceSampler


def build_data_loader(
        dataset: Union[List[Any], torchdata.Dataset],
        mapper: Callable[[Dict[str, Any]], Any],
        sampler: Optional[torchdata.Sampler] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        collate_fn: Optional[Callable[[List[Any]], Any]] = None,
) -> torchdata.DataLoader:
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        if sampler is None:
            sampler = InferenceSampler(len(dataset))
    return torchdata.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
    )


def build_ood_dataloader_args(cfg):
    """
    Builds the OOD dataset from a cfg file argument in cfg.PROBABILISTIC_INFERENCE.OOD_DATASET for OOD set.
    Assumes the dataset will be correctly found with this only argument
    :param cfg: Configuration class parameters
    :return: Dictionary of dataset, mapper, num_workers, sampler
    """
    dataset = get_detection_dataset_dicts(names=cfg.PROBABILISTIC_INFERENCE.OOD_DATASET,
                                          filter_empty=False,
                                          proposal_files=None,
                                          )
    mapper = DatasetMapper(cfg, False)

    return {
        "dataset": dataset,
        "mapper": mapper,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
        "sampler": InferenceSampler(len(dataset))
        if not isinstance(dataset, torchdata.IterableDataset)
        else None,
    }


def build_in_distribution_valid_test_dataloader_args(cfg,
                                                     dataset_name: str,
                                                     split_proportion: float) -> Tuple[Dict, Dict]:
    """
    Builds the arguments (datasets, mappers, samplers) for the validation and test sets starting form a single
    validation split set.
    :param cfg: Configuration class parameters
    :param dataset_name:
    :param split_proportion: Sets the proportion of the validation set
    :return: Tuple of two dictionaries to build the validation and test set dataloaders
    """
    if isinstance(dataset_name, str):
        dataset_name = [dataset_name]

    dataset = get_detection_dataset_dicts(
        dataset_name,
        filter_empty=False,
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(x)] for x in dataset_name
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )
    # Split dataset
    indexes_np = np.arange(len(dataset))
    np.random.shuffle(indexes_np)
    max_idx_valid = int(split_proportion * len(dataset))
    # valid_idxs, test_idxs = indexes_np[:12], indexes_np[100:115]
    valid_idxs, test_idxs = indexes_np[:max_idx_valid], indexes_np[max_idx_valid:]
    valid_dataset, test_dataset = [dataset[i] for i in valid_idxs], [dataset[i] for i in test_idxs]

    mapper_val = DatasetMapper(cfg, False)
    mapper_test = DatasetMapper(cfg, False)
    return {
        "dataset": valid_dataset,
        "mapper": mapper_val,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
        "sampler": InferenceSampler(len(valid_dataset))
        if not isinstance(valid_dataset, torchdata.IterableDataset)
        else None,
    }, {"dataset": test_dataset,
        "mapper": mapper_test,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
        "sampler": InferenceSampler(len(test_dataset))
        if not isinstance(test_dataset, torchdata.IterableDataset)
        else None, }


def build_ind_voc_train_dataloader_args(cfg,
                                        # dataset_name: str,
                                        split_proportion: float = 0.5) -> Dict:
    """
    Builds the arguments (datasets, mappers, samplers) for the VOC train set
    :param cfg: Configuration class parameters
    :param dataset_name:
    :param split_proportion: Sets the proportion of the train set
    :return: Dictionary to build the train set dataloader
    """
    # if isinstance(dataset_name, str):
    #     dataset_name = [dataset_name]

    # dataset = get_detection_dataset_dicts(
    #     dataset_name,
    #     filter_empty=False,
    #     proposal_files=[
    #         cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(x)] for x in dataset_name
    #     ]
    #     if cfg.MODEL.LOAD_PROPOSALS
    #     else None,
    # )
    dataset = get_detection_dataset_dicts(
        cfg.DATASETS.TRAIN,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
    )
    # _log_api_usage("dataset." + cfg.DATASETS.TRAIN[0])
    # Split dataset
    indexes_np = np.arange(len(dataset))
    np.random.shuffle(indexes_np)
    max_idx_train = int(split_proportion * len(dataset))
    # valid_idxs, test_idxs = indexes_np[:12], indexes_np[100:115]
    train_idxs = indexes_np[:max_idx_train]
    train_dataset = [dataset[i] for i in train_idxs]

    mapper_train = DatasetMapper(cfg, False)
    return {
        "dataset": train_dataset,
        "mapper": mapper_train,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
        "sampler": InferenceSampler(len(train_dataset))
        if not isinstance(train_dataset, torchdata.IterableDataset)
        else None,
    }
