import glob
import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import PIL
import PIL.Image
import torch
import pytorch_lightning as pl
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2


def anomaly_mud(pil_image,
                anomaly_width=2048,
                anomaly_height=1024,
                dirt_mud_images_path='./dirt_mud_images/'):
    
    anomaly_names = ['dirt1.png', 'mud1.png', 'mud2.png', 'mud3.png']
    anomaly_name = random.choice(anomaly_names)
    anomaly_mud_path = dirt_mud_images_path + anomaly_name
    mud_anomaly = PIL.Image.open(anomaly_mud_path)
    mud_anomaly = mud_anomaly.resize((anomaly_width, anomaly_height))
    mud_anomaly = mud_anomaly.convert("RGBA")

    # image = PIL.Image.open(test_img_path)
    img_failure = pil_image.copy()
    img_failure.paste(mud_anomaly, (0, 0), mud_anomaly)

    open_cv_image = np.array(img_failure)
    return open_cv_image


class AnomalyMud(ImageOnlyTransform):
    def __init__(self,
                 anomaly_width=2048,
                 anomaly_height=1024,
                 p=0.5,
                 dirt_mud_imgs_path='./dataset/dirt_mud_images/'):
        super().__init__(p=p)
        self.anomaly_width = anomaly_width
        self.anomaly_height = anomaly_height
        self.dirt_mud_imgs_path = dirt_mud_imgs_path

    def apply(self, img_cv, **params):
        pil_img = PIL.Image.fromarray(img_cv)
        return anomaly_mud(pil_img,
                           self.anomaly_width,
                           self.anomaly_height,
                           self.dirt_mud_imgs_path)


# Synthetic anomalies Unit-Test
def main():
    test_img_path = "./test_images/cityscapes_test_img.png"
    img_cv = cv2.cvtColor(cv2.imread(test_img_path), cv2.COLOR_BGR2RGB)
    # show clean image:
    # plt.figure()
    # plt.imshow(img_cv)

    synthetic_anomaly_transforms = A.Compose(
        [
            A.Resize(256, 512, p=1.0),
            A.OneOf([
                AnomalyMud(anomaly_width=512,
                           anomaly_height=256,
                           p=1,
                           dirt_mud_imgs_path='./dirt_mud_images/'),
                A.RandomFog(fog_coef_lower=0.7,
                            fog_coef_upper=0.8,
                            alpha_coef=0.3,
                            p=1),
                A.RandomSunFlare(flare_roi=(0.3, 0.1, 0.7, 0.9),
                                 src_radius=250,
                                 num_flare_circles_lower=6,
                                 num_flare_circles_upper=12,
                                 angle_lower=0.5,
                                 # angle_upper=0.5,
                                 p=1)
            ], p=1),
            # A.Normalize(mean=self.norm_mean, std=self.norm_std),
            # ToTensorV2()
        ]
    )
    transformed = synthetic_anomaly_transforms(image=img_cv)
    # show anomaly image:
    plt.figure()
    plt.imshow(transformed["image"])
    # show images:
    plt.show()


# Synthetic anomalies Unit-Test
if __name__ == "__main__":
    main()

