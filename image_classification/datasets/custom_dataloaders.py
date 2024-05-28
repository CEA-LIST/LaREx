import random
import torchvision
import torch
from icecream import ic
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms as transform_lib
from torch.utils.data.sampler import SubsetRandomSampler
from datasets import GtsrbModule, fmnist_to_gtsrb_format
from pl_bolts.datamodules import CIFAR10DataModule, FashionMNISTDataModule
from pl_bolts.datamodules import STL10DataModule
from .cifar10 import get_cifar10_input_transformations, fmnist_to_cifar_format


def get_data_loaders_image_classification(cfg, datasets_paths, n_workers):
    ##################################################################
    # GTSRB NORMAL DATASET
    ###################################################################
    ood_datasets_dict = {}
    if cfg.ind_dataset == "gtsrb":
        ic("gtsrb as InD")
        gtsrb_normal_dm = GtsrbModule(img_size=(cfg.datamodule.image_width, cfg.datamodule.image_height),
                                      data_path=datasets_paths["gtsrb"],
                                      # data_path=gtsrb_path,
                                      batch_size=1,
                                      shuffle=False, )

        gtsrb_normal_dm.setup(stage='fit')
        gtsrb_normal_dm.setup(stage='validate')
        gtsrb_normal_dm.setup(stage='test')

        # Subset train data loader to speed up OoD detection calculations
        gtsrb_ds_len = len(gtsrb_normal_dm.ds_gtsrb_train)
        indices_train_dl = list(range(gtsrb_ds_len))

        random.seed(cfg.seed)
        random.shuffle(indices_train_dl)

        split = int(np.floor(gtsrb_ds_len * cfg.train_subsamples_size))
        samples_idx = indices_train_dl[:split]
        # ic(len(samples_idx));

        train_sampler = SubsetRandomSampler(samples_idx)

        gtsrb_normal_dm.shuffle = False
        gtsrb_normal_dm.ds_gtsrb_train_sampler = train_sampler

        test_transforms = transform_lib.Compose([
            transform_lib.Resize((cfg.datamodule.image_width, cfg.datamodule.image_height)),
            transform_lib.ToTensor(),
            transform_lib.Normalize(
                mean=[0.3337, 0.3064, 0.3171],
                std=[0.2672, 0.2564, 0.2629]
            )
        ])

        ind_dataset_dict = {
            "train": gtsrb_normal_dm.train_dataloader(),
            "valid": gtsrb_normal_dm.val_dataloader(),
            "test": gtsrb_normal_dm.test_dataloader()
        }

    elif cfg.ind_dataset == "cifar10" and "gtsrb" in cfg.ood_datasets:
        ic("gtsrb as OoD")
        train_transforms, test_transforms = get_cifar10_input_transformations(
            cifar10_normalize_inputs=cfg.datamodule.cifar10_normalize_inputs,
            img_size=cfg.datamodule.image_width,
            data_augmentations="none",
            anomalies=False
        )
        gtsrb_normal_dm = GtsrbModule(img_size=(cfg.datamodule.image_width, cfg.datamodule.image_height),
                                      # data_path=gtsrb_path,
                                      data_path=datasets_paths["gtsrb"],
                                      batch_size=1,
                                      shuffle=False,
                                      train_transforms=train_transforms,
                                      valid_transforms=test_transforms,
                                      test_transforms=test_transforms,
                                      )
        gtsrb_normal_dm.setup(stage='fit')
        gtsrb_normal_dm.setup(stage='validate')
        gtsrb_normal_dm.setup(stage='test')
        # Add to ood datasets dict
        ood_datasets_dict["gtsrb"] = {
            "valid": gtsrb_normal_dm.val_dataloader(),
            "test": gtsrb_normal_dm.test_dataloader()
        }

    #####################################################################
    # GTSRB ANOMALIES DATASET
    #####################################################################
    if cfg.ind_dataset == "gtsrb" and "anomalies" in cfg.ood_datasets:
        ic("gtsrb anomalies as OoD")
        gtsrb_anomal_dm = GtsrbModule(
            img_size=(cfg.datamodule.image_width, cfg.datamodule.image_height),
            # data_path=gtsrb_path,
            data_path=datasets_paths["gtsrb"],
            batch_size=1,
            anomaly_transforms=True,
            shuffle=True,
        )

        gtsrb_anomal_dm.setup(stage='fit')
        gtsrb_anomal_dm.setup(stage='validate')
        gtsrb_anomal_dm.setup(stage='test')

        # Add to ood datasets dict
        ood_datasets_dict["gtsrb_anomal"] = {
            "valid": gtsrb_anomal_dm.val_dataloader(),
            "test": gtsrb_anomal_dm.test_dataloader()
        }

    ######################################################################
    # CIFAR10 DATASET
    ######################################################################
    if cfg.ind_dataset == "cifar10":
        ic("cifar10 as InD")
        train_transforms, test_transforms = get_cifar10_input_transformations(
            cifar10_normalize_inputs=cfg.datamodule.cifar10_normalize_inputs,
            img_size=cfg.datamodule.image_width,
            data_augmentations="none",
            anomalies=False
        )
        # cifar10_dm = CIFAR10DataModule(data_dir=cifar10_data_dir,
        cifar10_dm = CIFAR10DataModule(data_dir=datasets_paths["cifar10"],
                                       batch_size=1,
                                       train_transforms=train_transforms,
                                       test_transforms=test_transforms,
                                       val_transforms=test_transforms,
                                       )
        cifar10_dm.prepare_data()
        cifar10_dm.setup(stage='fit')
        cifar10_dm.setup(stage='test')
        # Subset train dataset
        subset_ds_len = int(len(cifar10_dm.dataset_train) * cfg.train_subsamples_size)
        cifar10_train_subset = torch.utils.data.random_split(
            cifar10_dm.dataset_train,
            [subset_ds_len, len(cifar10_dm.dataset_train) - subset_ds_len],
            torch.Generator().manual_seed(cfg.seed)
        )[0]
        # Subset the test dataset
        # Here, for several datasets a double split is needed since the script extracts by default from a
        # validation and test set
        cifar10_test_subset = torch.utils.data.random_split(
            cifar10_dm.dataset_test,
            [int(len(cifar10_dm.dataset_test) * cfg.datasets_sizes.cifar10),
             int(len(cifar10_dm.dataset_test) * (1.0 - cfg.datasets_sizes.cifar10))],
            torch.Generator().manual_seed(cfg.seed)
        )[0]
        cifar10_test_subset, cifar10_valid_subset = torch.utils.data.random_split(
            cifar10_test_subset,
            [int(len(cifar10_test_subset) * 0.5), int(len(cifar10_test_subset) * 0.5)],
            torch.Generator().manual_seed(cfg.seed)
        )

        cifar10_dm.shuffle = False
        ind_dataset_dict = {
            "train": DataLoader(cifar10_train_subset, batch_size=1, shuffle=True),
            "valid": DataLoader(cifar10_valid_subset, batch_size=1, shuffle=True),
            "test": DataLoader(cifar10_test_subset, batch_size=1, shuffle=True)
        }

    elif cfg.ind_dataset == "gtsrb" and "cifar10" in cfg.ood_datasets:
        ic("cifar10 as OoD")
        # cifar10_dm = CIFAR10DataModule(data_dir=cifar10_data_dir,
        cifar10_dm = CIFAR10DataModule(data_dir=datasets_paths["cifar10"],
                                       val_split=0.2,
                                       normalize=False,
                                       batch_size=1,
                                       seed=cfg.seed,
                                       drop_last=True,
                                       shuffle=True)

        cifar10_dm.train_transforms = test_transforms
        cifar10_dm.test_transforms = test_transforms
        cifar10_dm.val_transforms = test_transforms

        cifar10_dm.prepare_data()
        cifar10_dm.setup(stage='fit')
        cifar10_dm.setup(stage='test')

        # cifar10_train_loader = cifar10_dm.train_dataloader()
        ood_datasets_dict["cifar10"] = {
            "valid": cifar10_dm.val_dataloader(),
            "test": cifar10_dm.test_dataloader()
        }
    ######################################################################
    # CIFAR10 ANOMALIES DATASET
    ######################################################################
    if cfg.ind_dataset == "cifar10" and "anomalies" in cfg.ood_datasets:
        ic("cifar10 anomalies as OoD")
        anomal_train_transforms, anomal_test_transforms = get_cifar10_input_transformations(
            cifar10_normalize_inputs=cfg.datamodule.cifar10_normalize_inputs,
            img_size=cfg.datamodule.image_width,
            data_augmentations="none",
            anomalies=True,
        )
        cifar10_anomal_dm = CIFAR10DataModule(data_dir=datasets_paths["cifar10"],
                                              batch_size=1,
                                              train_transforms=anomal_train_transforms,
                                              test_transforms=anomal_test_transforms,
                                              val_transforms=anomal_test_transforms)
        cifar10_anomal_dm.prepare_data()
        cifar10_anomal_dm.setup(stage='fit')
        cifar10_anomal_dm.setup(stage='test')
        # Subset the test dataset
        cifar10_anomal_test_subset = torch.utils.data.random_split(
            cifar10_anomal_dm.dataset_test,
            [int(len(cifar10_anomal_dm.dataset_test) * cfg.datasets_sizes.cifar10),
             int(len(cifar10_anomal_dm.dataset_test) * (1.0 - cfg.datasets_sizes.cifar10))],
            torch.Generator().manual_seed(cfg.seed)
        )[0]
        cifar10_anomal_test_subset, cifar10_anomal_valid_subset = torch.utils.data.random_split(
            cifar10_anomal_test_subset,
            [int(len(cifar10_anomal_test_subset) * 0.5), int(len(cifar10_anomal_test_subset) * 0.5)],
            torch.Generator().manual_seed(cfg.seed)
        )
        ood_datasets_dict["cifar10_anomal"] = {
            "valid": DataLoader(cifar10_anomal_valid_subset, batch_size=1, shuffle=True),
            "test": DataLoader(cifar10_anomal_test_subset, batch_size=1, shuffle=True),
        }
    ##########################################################
    # STL-10 OoD
    #########################################################
    if "stl10" in cfg.ood_datasets:
        ic("stl10 as OoD")
        stl10_dm = STL10DataModule(data_dir=datasets_paths["stl10"],
                                   train_val_split=3000,
                                   num_workers=n_workers,
                                   batch_size=1,
                                   seed=cfg.seed,
                                   drop_last=True,
                                   shuffle=True)

        stl10_transforms = test_transforms

        stl10_dm.test_transforms = stl10_transforms
        stl10_dm.val_transforms = stl10_transforms

        stl10_dm.prepare_data()
        ood_datasets_dict["stl10"] = {
            "valid": stl10_dm.val_dataloader_labeled(),
            "test": stl10_dm.test_dataloader()
        }

    ##########################################################
    # Fashion MNIST OoD
    ##########################################################
    if "fmnist" in cfg.ood_datasets:
        ic("fmnist as OoD")
        fmnist_dm = FashionMNISTDataModule(data_dir=datasets_paths["fmnist"],
                                           val_split=0.2,
                                           num_workers=n_workers,
                                           normalize=False,
                                           batch_size=1,
                                           seed=cfg.seed,
                                           shuffle=True,
                                           drop_last=True
                                           )
        if cfg.ind_dataset == "cifar10":
            fmnist_transforms = fmnist_to_cifar_format(img_size=cfg.datamodule.image_width,
                                                       cifar_normalize=cfg.datamodule.cifar10_normalize_inputs)
        else:
            fmnist_transforms = fmnist_to_gtsrb_format(img_size=cfg.datamodule.image_width,
                                                       gtsrb_normalize=True)

        fmnist_dm.test_transforms = fmnist_transforms
        fmnist_dm.val_transforms = fmnist_transforms
        fmnist_dm.prepare_data()
        fmnist_dm.setup(stage='fit')
        fmnist_dm.setup(stage='test')
        # Subset test dataset
        fmnist_test_size = int((cfg.datasets_sizes.fmnist / 2) * len(fmnist_dm.dataset_test))
        fmnist_valid_size = int((cfg.datasets_sizes.fmnist / 2) * len(fmnist_dm.dataset_test))
        fmnist_test = torch.utils.data.random_split(
            fmnist_dm.dataset_test,
            [len(fmnist_dm.dataset_test) - fmnist_test_size - fmnist_valid_size,
             fmnist_test_size + fmnist_valid_size],
            torch.Generator().manual_seed(cfg.seed)
        )[1]
        fmnist_valid, fmnist_test = torch.utils.data.random_split(
            fmnist_test,
            [fmnist_valid_size, fmnist_test_size],
            torch.Generator().manual_seed(cfg.seed)
        )
        ood_datasets_dict["fmnist"] = {
            "valid": DataLoader(fmnist_valid, batch_size=1, shuffle=True),
            "test": DataLoader(fmnist_test, batch_size=1, shuffle=True),
        }
        del fmnist_dm
    ##########################################################
    # SVHN OoD
    ##########################################################
    if "svhn" in cfg.ood_datasets:
        ic("svhn as OoD")
        svhn_init_valid = torchvision.datasets.SVHN(
            root=datasets_paths["svhn"],
            split="test",
            download=True,
            transform=test_transforms
        )
        svhn_test_size = int((cfg.datasets_sizes.svhn / 2) * len(svhn_init_valid))
        svhn_valid_size = int((cfg.datasets_sizes.svhn / 2) * len(svhn_init_valid))
        svhn_test = torch.utils.data.random_split(
            svhn_init_valid,
            [len(svhn_init_valid) - svhn_valid_size - svhn_test_size, svhn_test_size + svhn_valid_size],
            torch.Generator().manual_seed(cfg.seed)
        )[1]
        svhn_valid, svhn_test = torch.utils.data.random_split(
            svhn_test,
            [svhn_valid_size, svhn_test_size],
            torch.Generator().manual_seed(cfg.seed)
        )
        # MNIST test set loader
        ood_datasets_dict["svhn"] = {
            "valid": DataLoader(svhn_valid, batch_size=1, shuffle=True),
            "test": DataLoader(svhn_test, batch_size=1, shuffle=True)
        }
        del svhn_init_valid
    ##########################################################
    # Places 365 OoD
    ##########################################################
    if "places" in cfg.ood_datasets:
        ic("places as OoD")
        places_init_valid = torchvision.datasets.Places365(datasets_paths["places"],
                                                           split="val",
                                                           small=True,
                                                           download=False,
                                                           transform=test_transforms, )
        places_test_size = int((cfg.datasets_sizes.places / 2) * len(places_init_valid))
        places_test = torch.utils.data.random_split(
            places_init_valid,
            [len(places_init_valid) - 2 * places_test_size, 2 * places_test_size],
            torch.Generator().manual_seed(cfg.seed)
        )[1]
        places_valid, places_test = torch.utils.data.random_split(
            places_test,
            [places_test_size, places_test_size],
            torch.Generator().manual_seed(cfg.seed)
        )
        # MNIST test set loader
        ood_datasets_dict["places"] = {
            "valid": DataLoader(places_valid, batch_size=1, shuffle=True),
            "test": DataLoader(places_test, batch_size=1, shuffle=True)
        }
        del places_init_valid

    ##########################################################
    # Textures OoD
    ##########################################################
    if "textures" in cfg.ood_datasets:
        ic("textures as OoD")
        textures_init_train = torchvision.datasets.DTD(datasets_paths["textures"],
                                                       split="train",
                                                       download=True,
                                                       transform=test_transforms, )
        textures_init_val = torchvision.datasets.DTD(datasets_paths["textures"],
                                                     split="val",
                                                     download=True,
                                                     transform=test_transforms, )
        textures_test = torchvision.datasets.DTD(datasets_paths["textures"],
                                                 split="test",
                                                 download=True,
                                                 transform=test_transforms, )
        textures_val = torch.utils.data.ConcatDataset([textures_init_train, textures_init_val])
        # MNIST test set loader
        ood_datasets_dict["textures"] = {
            "valid": DataLoader(textures_val, batch_size=1, shuffle=True),
            "test": DataLoader(textures_test, batch_size=1, shuffle=True)
        }

    ##########################################################
    # LSUN-Crop OoD
    ##########################################################
    if "lsun_c" in cfg.ood_datasets:
        print("LSUN-C as OoD")
        lsun_c_init_test = torchvision.datasets.ImageFolder(datasets_paths["lsun_c"], transform=test_transforms)
        lsun_c_test_subset = torch.utils.data.random_split(
            lsun_c_init_test,
            [int(len(lsun_c_init_test) * cfg.datasets_sizes.lsun_c),
             int(len(lsun_c_init_test) * (1.0 - cfg.datasets_sizes.lsun_c))],
            torch.Generator().manual_seed(cfg.seed)
        )[0]
        lsun_c_test_subset, lsun_c_valid_subset = torch.utils.data.random_split(
            lsun_c_test_subset,
            [int(len(lsun_c_test_subset) * 0.5), int(len(lsun_c_test_subset) * 0.5)],
            torch.Generator().manual_seed(cfg.seed)
        )
        ood_datasets_dict["lsun_c"] = {
            "valid": DataLoader(lsun_c_valid_subset, batch_size=1, shuffle=True),
            "test": DataLoader(lsun_c_test_subset, batch_size=1, shuffle=True),
        }
        del lsun_c_init_test

    ##########################################################
    # LSUN-Resize OoD
    ##########################################################
    if "lsun_r" in cfg.ood_datasets:
        print("LSUN-R as OoD")
        lsun_r_init_test = torchvision.datasets.ImageFolder(datasets_paths["lsun_r"], transform=test_transforms)
        lsun_r_test_subset = torch.utils.data.random_split(
            lsun_r_init_test,
            [int(len(lsun_r_init_test) * cfg.datasets_sizes.lsun_r),
             int(len(lsun_r_init_test) * (1.0 - cfg.datasets_sizes.lsun_r))],
            torch.Generator().manual_seed(cfg.seed)
        )[0]
        lsun_r_test_subset, lsun_r_valid_subset = torch.utils.data.random_split(
            lsun_r_test_subset,
            [int(len(lsun_r_test_subset) * 0.5), int(len(lsun_r_test_subset) * 0.5)],
            torch.Generator().manual_seed(cfg.seed)
        )
        ood_datasets_dict["lsun_r"] = {
            "valid": DataLoader(lsun_r_valid_subset, batch_size=1, shuffle=True),
            "test": DataLoader(lsun_r_test_subset, batch_size=1, shuffle=True),
        }
        del lsun_r_init_test

    ##########################################################
    # iSUN OoD
    ##########################################################
    if "isun" in cfg.ood_datasets:
        print("iSUN as OoD")
        isun_init_test = torchvision.datasets.ImageFolder(datasets_paths["isun"], transform=test_transforms)
        isun_test_subset = torch.utils.data.random_split(
            isun_init_test,
            [int(len(isun_init_test) * cfg.datasets_sizes.isun),
             len(isun_init_test) - (int(len(isun_init_test) * cfg.datasets_sizes.isun))],
            torch.Generator().manual_seed(cfg.seed)
        )[0]
        isun_test_subset, isun_valid_subset = torch.utils.data.random_split(
            isun_test_subset,
            [int(len(isun_test_subset) * 0.5), len(isun_test_subset) - int(len(isun_test_subset) * 0.5)],
            torch.Generator().manual_seed(cfg.seed)
        )
        ood_datasets_dict["isun"] = {
            "valid": DataLoader(isun_valid_subset, batch_size=1, shuffle=True),
            "test": DataLoader(isun_test_subset, batch_size=1, shuffle=True),
        }
        del isun_init_test

    return ind_dataset_dict, ood_datasets_dict
