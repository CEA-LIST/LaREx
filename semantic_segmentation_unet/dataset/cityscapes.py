import json
import os
from collections import namedtuple
from typing import Any, Callable
import torch
import torch.utils.data as data
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms as transform_lib
from PIL import Image
import numpy as np
from pytorch_lightning import LightningDataModule
import pytorch_lightning as pl
import torchmetrics
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .synthetic_anomalies import AnomalyMud
from icecream import ic
# from torchvision.datasets import Cityscapes


# from pytorch_lightning.utilities import _module_available
# _TORCHVISION_AVAILABLE: bool = _module_available("torchvision")
# if _TORCHVISION_AVAILABLE:
#     from torchvision import transforms as transform_lib
#     from torchvision.datasets import Cityscapes
# else:  # pragma: no cover
#     warn_missing_pkg("torchvision")


class Cityscapes(Dataset):
    """Cityscapes <http://www.cityscapes-dataset.com/> Dataset.

    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """
    # Code taken from https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/master/datasets/cityscapes.py
    # which is based on:
    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        CityscapesClass('unlabeled',            0,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle',          1,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi',           3,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static',               4,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic',              5,  255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground',               6,  255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road',                 7,  0,   'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk',             8,  1,   'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking',              9,  255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building',             11, 2,   'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall',                 12, 3,   'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence',                13, 4,   'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole',                 17, 5,   'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light',        19, 6,   'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign',         20, 7,   'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation',           21, 8,   'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain',              22, 9,   'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky',                  23, 10,  'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person',               24, 11,  'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider',                25, 12,  'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car',                  26, 13,  'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck',                27, 14,  'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus',                  28, 15,  'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train',                31, 16,  'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle',           32, 17,  'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle',              33, 18,  'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate',        -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])

    # train_id_to_color = [(0, 0, 0), (128, 64, 128), (70, 70, 70), (153, 153, 153), (107, 142, 35),
    #                      (70, 130, 180), (220, 20, 60), (0, 0, 142)]
    # train_id_to_color = np.array(train_id_to_color)
    # id_to_train_id = np.array([c.category_id for c in classes], dtype='uint8') - 1

    def __init__(self,
                 root,
                 split='train',
                 mode='fine',
                 target_type='semantic',
                 transform=None,
                 target_transform=None,
                 img_mask_transforms=None):

        self.root = os.path.expanduser(root)
        self.mode = 'gtFine'
        self.target_type = target_type
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)

        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.transform = transform
        self.target_transform = target_transform
        self.img_mask_transforms = img_mask_transforms

        self.split = split
        self.images = []
        self.targets = []

        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val"')

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)

            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                             self._get_target_suffix(self.mode, self.target_type))
                self.targets.append(os.path.join(target_dir, target_name))

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        # target[target == 255] = 19
        # target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

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

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])
        # image = image.resize((128, 256))
        # target = target.resize((128, 256))

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        if self.img_mask_transforms:
            transformed = self.img_mask_transforms(image=np.asarray(image),
                                                   mask=np.asarray(target))
            image = transformed['image']
            target = transformed['mask']
            target = target.unsqueeze(dim=0)

        target = self.encode_target(target)
        target[target == 255] = 19
        return image, target

    def __len__(self):
        return len(self.images)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        elif target_type == 'polygon':
            return '{}_polygons.json'.format(mode)
        elif target_type == 'depth':
            return '{}_disparity.png'.format(mode)


class CityscapesDataModule(LightningDataModule):
    """
    .. figure:: https://www.cityscapes-dataset.com/wordpress/wp-content/uploads/2015/07/muenster00-1024x510.png
        :width: 400
        :alt: Cityscape

    Standard Cityscapes, train, val, test splits and transforms

    Note: You need to have downloaded the Cityscapes dataset first and provide the path to where it is saved.
        You can download the dataset here: https://www.cityscapes-dataset.com/

    Specs:
        - 19 classes (road, person, sidewalk, etc...)
        - Original (image, target) - image dims: (3 x 1024 x 2048), target dims: (1024 x 2048)
        - User can define the size of both, images and target/labels
    """

    name = "Cityscapes Data Module"
    extra_args: dict = {}

    def __init__(
        self,
        data_dir: str,
        img_size: tuple = (128, 256),
        num_workers: int = 10,
        batch_size: int = 32,
        seed: int = 42,
        quality_mode: str = "fine",
        target_type: str = "semantic",
        norm_mean: list = None,
        norm_std: list = None,
        default_transforms: bool = False,
        default_img_mask_transforms: bool = False,
        img_transforms_train: transforms = None,
        img_transforms_valid: transforms = None,
        img_transforms_test: transforms = None,
        img_mask_transforms=None,
        shuffle: bool = False,
        pin_memory: bool = True,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: where to load the data from path, i.e. where directory leftImg8bit and gtFine or gtCoarse
                are located
            quality_mode: the quality mode to use, either 'fine' or 'coarse
            target_type: targets to use, either 'instance' or 'semantic'
            num_workers: how many workers to use for loading data
            batch_size: number of examples per training/eval step
            seed: random seed to be used for train/val/test splits
            shuffle: If true shuffles the data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """
        super().__init__(*args, **kwargs)

        if target_type not in ["instance", "semantic"]:
            raise ValueError(f'Only "semantic" and "instance" target types are supported. Got {target_type}.')

        self.dims = (3, 1024, 2048)
        self.img_size = img_size
        self.data_dir = data_dir
        self.quality_mode = quality_mode
        self.target_type = target_type
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.default_transforms = default_transforms
        self.default_img_mask_transforms = default_img_mask_transforms
        self.train_transforms = None
        self.val_transforms = None
        self.test_transforms = None
        self.target_transforms = None
        self.train_img_mask_transforms = None
        self.val_img_mask_transforms = None
        self.test_img_mask_transforms = None

        if norm_mean is None and norm_std is None:
            # self.norm_mean = [0.485, 0.456, 0.406]  # Common normalization values for mean
            # self.norm_std = [0.229, 0.224, 0.225]   # Common normalization values for std
            self.norm_mean = [0.28689554, 0.32513303, 0.28389177]  # Cityscapes normalization values mean
            self.norm_std = [0.18696375, 0.19017339, 0.18720214]  # Cityscapes normalizations values std
        else:
            self.norm_mean = norm_mean
            self.norm_std = norm_std

        if self.default_transforms:  # default transforms is True
            self.train_transforms = self._default_transforms()
            self.val_transforms = self._default_transforms()
            self.test_transforms = self._default_transforms()
            self.target_transforms = self._default_target_transforms()
        elif self.default_img_mask_transforms:  # Albumentations transforms!
            self.train_img_mask_transforms = self._train_default_img_mask_transforms()
            self.val_img_mask_transforms = self._val_default_img_mask_transforms()
            self.test_img_mask_transforms = self._val_default_img_mask_transforms()
        else:  # default transforms is False
            self.train_transforms = img_transforms_train
            self.val_transforms = img_transforms_valid
            self.test_transforms = img_transforms_test
            self.img_mask_transforms = img_mask_transforms
            self.target_transforms = self._default_target_transforms()

        self.save_hyperparameters()

    @property
    def num_classes(self) -> int:
        """
        Return:
            19
        """
        return 20

    def train_dataloader(self) -> DataLoader:
        """Cityscapes train set."""
        dataset = Cityscapes(
            root=self.data_dir,
            split="train",
            target_type=self.target_type,
            mode=self.quality_mode,
            transform=self.train_transforms,
            target_transform=self.target_transforms,
            img_mask_transforms=self.train_img_mask_transforms
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def val_dataloader(self) -> DataLoader:
        """Cityscapes val set."""
        dataset = Cityscapes(
            root=self.data_dir,
            split="val",
            target_type=self.target_type,
            mode=self.quality_mode,
            transform=self.val_transforms,
            target_transform=self.target_transforms,
            img_mask_transforms=self.val_img_mask_transforms
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )
        return loader

    def test_dataloader(self) -> DataLoader:
        """Cityscapes test set."""
        dataset = Cityscapes(
            root=self.data_dir,
            split="test",
            target_type=self.target_type,
            mode=self.quality_mode,
            transform=self.test_transforms,
            target_transform=self.target_transforms,
            img_mask_transforms=self.test_img_mask_transforms
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def anomaly_val_dataloader(self) -> DataLoader:
        """Cityscapes val set."""
        dataset = Cityscapes(
            root=self.data_dir,
            split="val",
            target_type=self.target_type,
            mode=self.quality_mode,
            transform=None,
            target_transform=None,
            img_mask_transforms=self._anomaly_transforms()
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )
        return loader

    def anomaly_test_dataloader(self) -> DataLoader:
        """Cityscapes test set."""
        dataset = Cityscapes(
            root=self.data_dir,
            split="test",
            target_type=self.target_type,
            mode=self.quality_mode,
            transform=None,
            target_transform=None,
            img_mask_transforms=self._anomaly_transforms()
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def _default_transforms(self) -> Callable:
        cityscapes_img_transforms = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.norm_mean, std=self.norm_std)
            ])
        return cityscapes_img_transforms

    def _default_target_transforms(self) -> Callable:
        cityscapes_target_transforms = transform_lib.Compose(
            [transform_lib.Resize(self.img_size),
             transform_lib.PILToTensor()])
        return cityscapes_target_transforms

    def _train_default_img_mask_transforms(self) -> Callable:
        cityscapes_train_img_mask_transforms = A.Compose(
            [
                A.Resize(self.img_size[0], self.img_size[1], p=1.0),
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

        return cityscapes_train_img_mask_transforms

    def _val_default_img_mask_transforms(self) -> Callable:
        cityscapes_val_img_mask_transforms = A.Compose(
            [
                A.Resize(self.img_size[0], self.img_size[1]),
                A.Normalize(mean=self.norm_mean, std=self.norm_std),
                ToTensorV2()
            ]
        )

        return cityscapes_val_img_mask_transforms

    def _anomaly_transforms(self) -> Callable:
        cityscapes_anomaly_transforms = A.Compose(
            [
                A.Resize(self.img_size[0], self.img_size[1], p=1),
                A.OneOf([
                    AnomalyMud(anomaly_width=self.img_size[1],
                               anomaly_height=self.img_size[0],
                               p=1),
                    A.RandomFog(fog_coef_lower=0.7, fog_coef_upper=0.8, alpha_coef=0.7, p=1),
                    A.RandomSunFlare(flare_roi=(0.2, 0.1, 0.8, 0.9),
                                     src_radius=int(self.img_size[1] * 0.8),
                                     num_flare_circles_lower=8,
                                     num_flare_circles_upper=12,
                                     angle_lower=0.5,
                                     p=1)
                ], p=1),
                A.Normalize(mean=self.norm_mean, std=self.norm_std),
                ToTensorV2()
            ]
        )
        return cityscapes_anomaly_transforms


# Cityscapes Unit-Test
def main():
    dataset_path = '/media/farnez/Data/DATASETS/CityScapes'
    batch_size = 16
    img_h = 128
    img_w = 256

    cityscapes_input_transforms = transform_lib.Compose([
        transform_lib.Resize((img_h, img_w)),
        transform_lib.ToTensor(),
        transform_lib.Normalize(
            mean=[0.28689554, 0.32513303, 0.28389177],
            std=[0.18696375, 0.19017339, 0.18720214]
        )
    ])

    cityscapes_target_transforms = transform_lib.Compose(
        [transform_lib.Resize((img_h, img_w)),
         transform_lib.PILToTensor()])

    cs_train_ds = Cityscapes(root=dataset_path,
                             split='train',
                             mode='gtFine',
                             transform=cityscapes_input_transforms,
                             target_transform=cityscapes_target_transforms)

    cs_train_loader = DataLoader(cs_train_ds,
                                 batch_size=batch_size,
                                 num_workers=10,
                                 drop_last=True)

    for img, mask in cs_train_loader:

        print(img.shape)
        print(mask.shape)


# Cityscapes Unit-Test
if __name__ == "__main__":
    main()
