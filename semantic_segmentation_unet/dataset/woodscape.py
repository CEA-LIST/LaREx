from typing import Optional, Any, Callable
import os
import glob
import torch
import random
from torch.utils.data import Dataset 
import torchvision
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import PIL
import PIL.Image
import numpy as np
from pytorch_lightning import LightningDataModule
import pytorch_lightning as pl
import torchmetrics
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .synthetic_anomalies import AnomalyMud
from icecream import ic


class WoodScapeDataset(Dataset):
    def __init__(self,
                 dataset_dir: str,
                 img_size: tuple = (150, 300),
                 n_classes: int = 10,
                 default_transforms=False,
                 img_transforms=None,
                 label_transforms=None,
                 img_mask_transforms=None,
                 label_colours=None) -> None:
        super().__init__()
        self.class_names = ["void",
                            "road",
                            "lanemarks",
                            "curb",
                            "person",
                            "rider",
                            "vehicles",
                            "bicycle",
                            "motorcycle",
                            "traffic_sign"]

        self.n_classes = n_classes

        self.class_colors_rgb = [
            [0, 0, 0],
            [255, 0, 255],
            [0, 0, 255],
            [0, 255, 0],
            [255, 0, 0],
            [255, 255, 255],
            [0, 255, 255],
            [255, 255, 0],
            [255, 128, 128],
            [128, 128, 0]
        ]  # RGB format!

        self.class_colors_bgr = [
            [0, 0, 0],
            [255, 0, 255],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 255],
            [255, 255, 0],
            [0, 255, 255],
            [128, 128, 255],
            [0, 128, 128]
        ]  # BGR format!

        if label_colours is None:
            self.label_colours = dict(zip(range(10), self.class_colors_rgb))

        else:
            self.label_colours = label_colours

        self.class_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        self.dataset_dir = dataset_dir
        self.rgb_images_dir = os.path.join(self.dataset_dir, "rgb_images")
        self.rgb_images_list = glob.glob(os.path.join(self.rgb_images_dir, "*.png"))
        self.rgb_images_list.sort()
        self.semantic_annotations_dir = os.path.join(self.dataset_dir, "semantic_annotations")
        self.semantic_annotations_gt_dir = os.path.join(self.semantic_annotations_dir, "gtLabels")
        self.semantic_annotations_rgb_dir = os.path.join(self.semantic_annotations_dir, "rgbLabels")
        self.semantic_annotations_gt_list = glob.glob(os.path.join(self.semantic_annotations_gt_dir, "*.png"))
        self.semantic_annotations_gt_list.sort()
        self.semantic_annotations_rgb_list = glob.glob(os.path.join(self.semantic_annotations_rgb_dir, "*.png"))
        self.semantic_annotations_rgb_list.sort()

        self.img_size = img_size
        if self.img_size is None:
            self.img_size = (150, 300)

        self.default_transforms = default_transforms
        self.img_mask_transforms = img_mask_transforms

        if self.default_transforms:  # default transforms is True
            self.img_transforms = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.32757252, 0.33050337, 0.33689716],
                    std=[0.20290555, 0.20525302, 0.2037721]
                )
            ])
            self.label_transforms = transforms.Compose([
                transforms.Resize(self.img_size)])

        else:  # default transforms is False
            self.img_transforms = img_transforms
            self.label_transforms = label_transforms

    def __len__(self):
        """__len__"""
        return len(self.rgb_images_list)

    def __getitem__(self, index):
        # get image path
        img_path = self.rgb_images_list[index]
        gt_label_path = self.semantic_annotations_gt_list[index]

        image = PIL.Image.open(img_path).convert('RGB')
        gt_label = PIL.Image.open(gt_label_path)

        # starts here 1:
        # image = self.img_transforms(image)
        # image = image.float()
        #
        # gt_label_np = np.asarray(gt_label)
        # gt_label_tensor = torch.LongTensor(np.array(gt_label_np, copy=True))
        # gt_label_tensor = gt_label_tensor.unsqueeze(0)
        # gt_label_tensor = self.label_transforms(gt_label_tensor)

        # starts here 2:
        if self.img_transforms:
            image = self.img_transforms(image)
            image = image.float()

        if self.label_transforms:
            gt_label = np.asarray(gt_label)
            gt_label = torch.LongTensor(np.array(gt_label, copy=True))
            gt_label = gt_label.unsqueeze(0)
            gt_label = self.label_transforms(gt_label)

        if self.img_mask_transforms:
            transformed = self.img_mask_transforms(image=np.asarray(image),
                                                   mask=np.asarray(gt_label))
            image = transformed['image']
            gt_label = transformed['mask']
            gt_label = gt_label.unsqueeze(dim=0)

        # return image, gt_label_tensor
        return image, gt_label

    def decode_segmap(self, temp: np.ndarray):
        temp = temp.squeeze()
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb


class WoodScapeDataModule(LightningDataModule):
    name = "WoodScape"
    extra_args: dict = {}

    def __init__(self,
                 dataset_dir: str,
                 target_type: str = 'semantic',
                 num_workers: int = 10,
                 batch_size: int = 32,
                 valid_size: float = 0.2,
                 test_size: float = 0.1,
                 seed: int = 10,
                 img_size: tuple = (150, 300),
                 shuffle: bool = False,
                 pin_memory: bool = True,
                 drop_last: bool = False,
                 default_transforms: bool = False,
                 default_img_mask_transforms: bool = False,
                 img_transforms_train: transforms = None,
                 img_transforms_valid: transforms = None,
                 label_colours: dict = None,
                 norm_mean: list = None,
                 norm_std: list = None,
                 *args: Any,
                 **kwargs: Any,
                 ):
        super().__init__(*args, **kwargs)

        self.dataset_dir = dataset_dir
        self.target_type = target_type
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.test_size = test_size
        self.default_transforms = default_transforms
        self.default_img_mask_transforms = default_img_mask_transforms
        self.label_transforms = None
        self.seed = seed
        self.img_size = img_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.woodscape_ds_train = None
        self.woodscape_ds_valid = None
        self.woodscape_anomaly_ds = None
        self.train_sampler = None
        self.valid_sampler = None
        self.test_sampler = None

        self.label_colours = label_colours

        self.train_loader_len = 0
        self.valid_loader_len = 0
        self.test_loader_len = 0
        self.anomaly_valid_loader_len = 0
        self.anomaly_test_loader_len = 0

        if norm_mean is None and norm_std is None:
            self.norm_mean = [0.32757252, 0.33050337, 0.33689716]
            self.norm_std = [0.20290555, 0.20525302, 0.2037721]
        else:
            self.norm_mean = norm_mean
            self.norm_std = norm_std

        if self.default_transforms:  # default transforms is True
            self.img_transforms_train = self._default_img_transforms()
            self.img_transforms_valid = self._default_img_transforms()
            self.label_transforms = self._default_label_transforms()

        elif self.default_img_mask_transforms:
            self.img_transforms_train = self._train_default_img_mask_transforms()
            self.img_transforms_valid = self._val_default_img_mask_transforms()

        else:  # default transforms is False
            self.img_transforms_train = img_transforms_train
            self.img_transforms_valid = img_transforms_valid
            # self.label_transforms_train = label_transforms_train
            # self.label_transforms_train = img_transforms_train
            self.label_transforms = self._default_label_transforms()
            # self.label_transforms_valid = label_transforms_valid

        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.woodscape_ds_train = WoodScapeDataset(dataset_dir=self.dataset_dir,
                                                       img_size=self.img_size,
                                                       img_transforms=self.img_transforms_train,
                                                       label_transforms=self.label_transforms,
                                                       label_colours=self.label_colours)

            self.woodscape_ds_valid = WoodScapeDataset(dataset_dir=self.dataset_dir,
                                                       img_size=self.img_size,
                                                       img_transforms=self.img_transforms_valid,
                                                       label_transforms=self.label_transforms,
                                                       label_colours=self.label_colours)

            self.woodscape_anomaly_ds = WoodScapeDataset(dataset_dir=self.dataset_dir,
                                                         img_size=self.img_size,
                                                         img_transforms=None,
                                                         label_transforms=None,
                                                         img_mask_transforms=self._anomaly_transforms(),
                                                         label_colours=self.label_colours)

            # get dataset length (woodscape_ds_train and woodscape_ds_valid are the same):
            dataset_len = len(self.woodscape_ds_train)
            # get indices
            indices = list(range(dataset_len))
            # indices random shuffle
            print("DATASET Shuffle Random SEED: ", self.seed)
            random.seed(self.seed)
            random.shuffle(indices)
            # split dataset into train subset and test set:
            split = int(np.floor(self.test_size * dataset_len))
            train_ss_idx, test_idx = indices[split:], indices[:split]
            # split train subset into train set and validation set:
            train_set_len = len(train_ss_idx)
            split_train = int(np.floor(self.valid_size * train_set_len))
            train_idx, valid_idx = train_ss_idx[split_train:], train_ss_idx[:split_train]
            # define samplers for obtaining training and validation batches
            self.train_sampler = SubsetRandomSampler(train_idx)
            self.valid_sampler = SubsetRandomSampler(valid_idx)  # Sampler is useful for woodscape_ds_valid
            self.test_sampler = SubsetRandomSampler(test_idx)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            pass
            # dataset_test = WoodScapeDataset(dataset_dir=self.dataset_dir)
            # TODO: add test set when available from data provider

    def train_dataloader(self) -> DataLoader:
        woodscape_train_loader = DataLoader(self.woodscape_ds_train,
                                            batch_size=self.batch_size,
                                            sampler=self.train_sampler,
                                            num_workers=self.num_workers,
                                            pin_memory=self.pin_memory,
                                            drop_last=self.drop_last)
        self.train_loader_len = len(woodscape_train_loader)
        return woodscape_train_loader

    def val_dataloader(self) -> DataLoader:
        woodscape_valid_loader = DataLoader(self.woodscape_ds_valid,
                                            batch_size=self.batch_size,
                                            sampler=self.valid_sampler,
                                            num_workers=self.num_workers,
                                            pin_memory=self.pin_memory,
                                            drop_last=self.drop_last)
        self.valid_loader_len = len(woodscape_valid_loader)
        return woodscape_valid_loader

    def test_dataloader(self) -> DataLoader:
        # TODO: add test dataloader when available from data provider
        woodscape_test_loader = DataLoader(self.woodscape_ds_train,
                                           batch_size=self.batch_size,
                                           sampler=self.test_sampler,
                                           num_workers=self.num_workers,
                                           pin_memory=self.pin_memory,
                                           drop_last=self.drop_last)
        self.test_loader_len = len(woodscape_test_loader)
        return woodscape_test_loader

    def anomaly_val_dataloader(self) -> DataLoader:
        ws_anomaly_valid_loader = DataLoader(self.woodscape_anomaly_ds,
                                             batch_size=self.batch_size,
                                             sampler=self.valid_sampler,
                                             num_workers=self.num_workers,
                                             pin_memory=self.pin_memory,
                                             drop_last=self.drop_last)
        self.anomaly_valid_loader_len = len(ws_anomaly_valid_loader)
        return ws_anomaly_valid_loader

    def anomaly_test_dataloader(self) -> DataLoader:
        # TODO: add test dataloader when available from data provider
        ws_anomaly_test_loader = DataLoader(self.woodscape_anomaly_ds,
                                            batch_size=self.batch_size,
                                            sampler=self.test_sampler,
                                            num_workers=self.num_workers,
                                            pin_memory=self.pin_memory,
                                            drop_last=self.drop_last)
        self.anomaly_test_loader_len = len(ws_anomaly_test_loader)
        return ws_anomaly_test_loader

    def _default_img_transforms(self):
        woodscape_img_transforms = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.norm_mean, std=self.norm_std)
            ])
        return woodscape_img_transforms

    def _default_label_transforms(self):
        woodscape_label_transforms = transforms.Compose([
                transforms.Resize(self.img_size)])
        return woodscape_label_transforms

    def _train_default_img_mask_transforms(self) -> Callable:
        woodscape_train_img_mask_transforms = A.Compose(
            [
                A.Resize(self.img_size[0], self.img_size[1], p=1.0),
                # A.OneOf([
                #     A.Resize(self.img_size[0], self.img_size[1], p=1.0),
                #     A.RandomResizedCrop(self.img_size[0], self.img_size[1], p=1.0),
                # ], p=1.0),
                # A.Resize(self.img_size[0], self.img_size[1]),
                # # A.RandomResizedCrop(self.img_size[0], self.img_size[1], p=1),
                A.OneOf([
                    A.RandomResizedCrop(self.img_size[0], self.img_size[1], p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.HueSaturationValue(p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                ], p=0.5),
                A.Normalize(mean=self.norm_mean, std=self.norm_std),
                ToTensorV2()
            ]
        )

        return woodscape_train_img_mask_transforms

    def _val_default_img_mask_transforms(self) -> Callable:
        woodscape_val_img_mask_transforms = A.Compose(
            [
                A.Resize(self.img_size[0], self.img_size[1]),
                A.Normalize(mean=self.norm_mean, std=self.norm_std),
                ToTensorV2()
            ]
        )

        return woodscape_val_img_mask_transforms

    def _anomaly_transforms(self) -> Callable:
        woodscape_anomaly_transforms = A.Compose(
            [
                A.Resize(self.img_size[0], self.img_size[1], p=1),
                A.OneOf([
                    AnomalyMud(anomaly_width=self.img_size[1],
                               anomaly_height=self.img_size[0],
                               p=1),
                    A.RandomFog(fog_coef_lower=0.7, fog_coef_upper=0.8, alpha_coef=0.6, p=1),
                    A.RandomSunFlare(flare_roi=(0.2, 0.2, 0.8, 0.8),
                                     src_radius=int(self.img_size[1] * 0.7),
                                     num_flare_circles_lower=6,
                                     num_flare_circles_upper=12,
                                     angle_lower=0.5,
                                     p=1)
                ], p=1),
                A.Normalize(mean=self.norm_mean, std=self.norm_std),
                ToTensorV2()
            ]
        )
        return woodscape_anomaly_transforms


class WoodScapeSoilingDataset(Dataset):
    def __init__(self,
                 dataset_dir: str,
                 train: bool = True,
                 img_size: tuple = (150, 300),
                 default_transforms=False,
                 img_transforms=None,
                 label_transforms=None) -> None:
        super().__init__()
        self.class_names = ["clear",
                            "transparent",
                            "semi_transparent",
                            "opaque"]

        self.n_classes = 4

        self.class_colors_rgb = [
            [0, 0, 0],
            [0, 255, 0],
            [255, 0, 0],
            [0, 0, 255]
        ]  # RGB format!

        self.class_colors_bgr = [
            [0, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 0, 0]
        ]  # BGR format!

        self.label_colours = dict(zip(range(4), self.class_colors_rgb))
        self.class_indexes = [0, 1, 2, 3]

        self.dataset_dir = dataset_dir
        self.train = train

        if self.train:
            self.dataset_dir = os.path.join(self.dataset_dir, "train")
        else:
            self.dataset_dir = os.path.join(self.dataset_dir, "test")

        self.rgb_soil_images_dir = os.path.join(self.dataset_dir, "rgbImages")
        self.rgb_soil_images_list = glob.glob(os.path.join(self.rgb_soil_images_dir, "*.png"))
        self.rgb_soil_images_list.sort()
        ic(len(self.rgb_soil_images_list))

        self.soil_annotations_gt_dir = os.path.join(self.dataset_dir, "gtLabels")
        self.soil_annotations_gt_list = glob.glob(os.path.join(self.soil_annotations_gt_dir, "*.png"))
        self.soil_annotations_gt_list.sort()
        ic(len(self.soil_annotations_gt_list))

        self.soil_annotations_rgb_dir = os.path.join(self.dataset_dir, "rgbLabels")
        self.soil_annotations_rgb_list = glob.glob(os.path.join(self.soil_annotations_rgb_dir, "*.png"))
        self.soil_annotations_rgb_list.sort()
        ic(len(self.soil_annotations_rgb_list))

        self.img_size = img_size
        if self.img_size is None:
            self.img_size = (150, 300)

        self.default_transforms = default_transforms

        if self.default_transforms:
            self.img_transforms = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.32757252, 0.33050337, 0.33689716],
                    std=[0.20290555, 0.20525302, 0.2037721]
                )
            ])
            self.label_transforms = transforms.Compose([
                transforms.Resize(self.img_size)])
        else:
            self.img_transforms = img_transforms
            self.label_transforms = label_transforms

    def __len__(self):
        """__len__"""
        return len(self.rgb_soil_images_list)

    def __getitem__(self, index):
        # get image path
        img_path = self.rgb_soil_images_list[index]
        gt_label_path = self.soil_annotations_gt_list[index]

        image = PIL.Image.open(img_path).convert('RGB')
        gt_label = PIL.Image.open(gt_label_path)

        image = self.img_transforms(image)
        image = image.float()

        gt_label_np = np.asarray(gt_label)
        gt_label_tensor = torch.LongTensor(np.array(gt_label_np, copy=True))
        gt_label_tensor = gt_label_tensor.unsqueeze(0)
        gt_label_tensor = self.label_transforms(gt_label_tensor)

        return image, gt_label_tensor

    def decode_segmap(self, temp: np.ndarray):
        temp = temp.squeeze()
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb


class WoodScapeSoilingDataModule(LightningDataModule):
    name = "WoodScape-Soiling"
    extra_args: dict = {}

    def __init__(self,
                 dataset_dir: str,
                 num_workers: int = 10,
                 batch_size: int = 32,
                 valid_size: float = 0.2,
                 seed: int = 10,
                 img_size: tuple = (150, 300),
                 shuffle: bool = False,
                 pin_memory: bool = True,
                 drop_last: bool = False,
                 img_transforms: transforms = None,
                 label_transforms: transforms = None,
                 default_transforms: bool = False,
                 *args: Any,
                 **kwargs: Any,
                 ):
        super().__init__(*args, **kwargs)
        self.dataset_dir = dataset_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.default_transforms = default_transforms
        self.seed = seed
        self.img_size = img_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.woodscape_soil_ds_train = None
        self.woodscape_soil_ds_valid = None
        self.woodscape_soil_ds_test = None
        self.train_sampler = None
        self.valid_sampler = None

        if self.default_transforms:
            self.img_transforms = self._default_img_transforms()
            self.label_transforms = self._default_label_transforms()
        else:
            self.img_transforms = img_transforms
            self.label_transforms = label_transforms

    def setup(self, stage: Optional[str] = None) -> None:
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.woodscape_soil_ds_train = WoodScapeSoilingDataset(dataset_dir=self.dataset_dir,
                                                                   train=True,
                                                                   img_size=self.img_size,
                                                                   img_transforms=self.img_transforms,
                                                                   label_transforms=self.label_transforms)
            # get dataset length
            dataset_train_len = len(self.woodscape_soil_ds_train)
            ic(dataset_train_len)

            # get indices
            indices = list(range(dataset_train_len))
            # indices random shuffle
            print("DATASET Shuffle Random SEED: ", self.seed)
            random.seed(self.seed)
            random.shuffle(indices)
            # split
            split = int(np.floor(self.valid_size * dataset_train_len))
            train_idx, valid_idx = indices[split:], indices[:split]
            # define samplers for obtaining training and validation batches
            self.train_sampler = SubsetRandomSampler(train_idx)
            self.valid_sampler = SubsetRandomSampler(valid_idx)
        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.woodscape_soil_ds_test = WoodScapeSoilingDataset(dataset_dir=self.dataset_dir,
                                                                  train=False,
                                                                  img_size=self.img_size,
                                                                  img_transforms=self.img_transforms,
                                                                  label_transforms=self.label_transforms)
            dataset_test_len = len(self.woodscape_soil_ds_test)
            ic(dataset_test_len)

    def train_dataloader(self) -> DataLoader:
        woodscape_soil_train_loader = DataLoader(self.woodscape_soil_ds_train,
                                                 batch_size=self.batch_size,
                                                 sampler=self.train_sampler,
                                                 num_workers=self.num_workers,
                                                 pin_memory=self.pin_memory,
                                                 drop_last=self.drop_last)
        return woodscape_soil_train_loader

    def val_dataloader(self) -> DataLoader:
        woodscape_soil_valid_loader = DataLoader(self.woodscape_soil_ds_train,
                                                 batch_size=self.batch_size,
                                                 sampler=self.valid_sampler,
                                                 num_workers=self.num_workers,
                                                 pin_memory=self.pin_memory,
                                                 drop_last=self.drop_last)
        return woodscape_soil_valid_loader

    def test_dataloader(self) -> DataLoader:
        woodscape_soil_test_loader = DataLoader(self.woodscape_soil_ds_test,
                                                batch_size=self.batch_size,
                                                num_workers=self.num_workers,
                                                pin_memory=self.pin_memory,
                                                drop_last=self.drop_last)
        return woodscape_soil_test_loader

    def _default_img_transforms(self):
        woodscape_img_transforms = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.32757252, 0.33050337, 0.33689716],
                    std=[0.20290555, 0.20525302, 0.2037721]
                )
            ])
        return woodscape_img_transforms

    def _default_label_transforms(self):
        woodscape_label_transforms = transforms.Compose([
                transforms.Resize(self.img_size)])
        return woodscape_label_transforms


class WoodScapeOoDBenchmarkDataModule(LightningDataModule):
    name = "OoD-Benchmark"

    def __init__(self,
                 dataset_dir: str = '/media/farnez/Data/DATASETS/WoodScape/monitor_benchmark/',
                 num_workers: int = 10,
                 batch_size: int = 32,
                 seed: int = 10,
                 img_size: tuple = (128, 128),
                 shuffle: bool = False,
                 pin_memory: bool = True,
                 drop_last: bool = False,
                 img_transforms: transforms = None,
                 default_transforms: bool = True,
                 *args: Any,
                 **kwargs: Any,
                 ):
        super().__init__(*args, **kwargs)
        self.dataset_dir = dataset_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.img_size = img_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.default_transforms = default_transforms
        self.ds_normal = None
        self.ds_shift = None
        self.ds_anomaly = None
        self.ds_normal_shift = None
        self.ds_normal_anomaly = None
        self.ds_benchmark_train = None
        self.ds_benchmark_test = None

        if self.default_transforms:
            self.img_transforms = self._default_img_transforms()

    def setup(self, stage: Optional[str] = None) -> None:
        ds_normal_path = "/media/farnez/Data/DATASETS/WoodScape/monitor_benchmark/ds_normal"
        self.ds_normal = torchvision.datasets.ImageFolder(ds_normal_path,
                                                          transform=self.img_transforms)

        ds_shift_path = "/media/farnez/Data/DATASETS/WoodScape/monitor_benchmark/ds_shift"
        self.ds_shift = torchvision.datasets.ImageFolder(ds_shift_path,
                                                         transform=self.img_transforms)

        ds_anomaly_path = "/media/farnez/Data/DATASETS/WoodScape/monitor_benchmark/ds_anomaly"
        self.ds_anomaly = torchvision.datasets.ImageFolder(ds_anomaly_path,
                                                           transform=self.img_transforms)

        ds_normal_shift_path = "/media/farnez/Data/DATASETS/WoodScape/monitor_benchmark/ds_normal_shift"
        self.ds_normal_shift = torchvision.datasets.ImageFolder(ds_normal_shift_path,
                                                                transform=self.img_transforms)

        ds_normal_anomaly_path = "/media/farnez/Data/DATASETS/WoodScape/monitor_benchmark/ds_normal_anomaly"
        self.ds_normal_anomaly = torchvision.datasets.ImageFolder(ds_normal_anomaly_path,
                                                                  transform=self.img_transforms)

        ds_benchmark_train_path = "/media/farnez/Data/DATASETS/WoodScape/monitor_benchmark/ds_benchmark/train"
        self.ds_benchmark_train = torchvision.datasets.ImageFolder(ds_benchmark_train_path,
                                                                   transform=self.img_transforms)

        ds_benchmark_test_path = "/media/farnez/Data/DATASETS/WoodScape/monitor_benchmark/ds_benchmark/test"
        self.ds_benchmark_test = torchvision.datasets.ImageFolder(ds_benchmark_test_path,
                                                                  transform=self.img_transforms)

    def get_ds_normal_loader(self):
        ds_normal_loader = DataLoader(self.ds_normal,
                                      batch_size=self.batch_size,
                                      num_workers=self.num_workers,
                                      pin_memory=self.pin_memory,
                                      drop_last=self.drop_last)

        return ds_normal_loader

    def get_ds_shift_loader(self):
        ds_shift_loader = DataLoader(self.ds_shift,
                                     batch_size=self.batch_size,
                                     num_workers=self.num_workers,
                                     pin_memory=self.pin_memory,
                                     drop_last=self.drop_last)

        return ds_shift_loader

    def get_ds_anomaly_loader(self):
        ds_anomaly_loader = DataLoader(self.ds_anomaly,
                                       batch_size=self.batch_size,
                                       num_workers=self.num_workers,
                                       pin_memory=self.pin_memory,
                                       drop_last=self.drop_last)

        return ds_anomaly_loader

    def get_ds_normal_shift_loader(self):
        ds_normal_shift_loader = DataLoader(self.ds_normal_shift,
                                            batch_size=self.batch_size,
                                            num_workers=self.num_workers,
                                            pin_memory=self.pin_memory,
                                            drop_last=self.drop_last)

        return ds_normal_shift_loader

    def get_ds_normal_anomaly_loader(self):
        ds_normal_anomaly_loader = DataLoader(self.ds_normal_anomaly,
                                              batch_size=self.batch_size,
                                              num_workers=self.num_workers,
                                              pin_memory=self.pin_memory,
                                              drop_last=self.drop_last)

        return ds_normal_anomaly_loader

    def get_ds_benchmark_train_loader(self):
        ds_benchmark_train_loader = DataLoader(self.ds_benchmark_train,
                                               batch_size=self.batch_size,
                                               num_workers=self.num_workers,
                                               pin_memory=self.pin_memory,
                                               drop_last=self.drop_last)

        return ds_benchmark_train_loader

    def get_ds_benchmark_test_loader(self):
        ds_benchmark_test_loader = DataLoader(self.ds_benchmark_test,
                                              batch_size=self.batch_size,
                                              num_workers=self.num_workers,
                                              pin_memory=self.pin_memory,
                                              drop_last=self.drop_last)

        return ds_benchmark_test_loader

    def _default_img_transforms(self):
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]
        # norm_mean = [0.32757252, 0.33050337, 0.33689716]
        # norm_std = [0.20290555, 0.20525302, 0.2037721]
        woodscape_img_transforms = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=norm_mean,
                    std=norm_std
                )
            ])
        return woodscape_img_transforms


# Woodscape Unit-Test
def main():
    woodscape_ds = WoodScapeDataset(dataset_dir="/media/farnez/Data/DATASETS/WoodScape/")
    print(woodscape_ds.rgb_images_list)
    pass


# Woodscape Unit-Test
if __name__ == "__main__":
    main()
