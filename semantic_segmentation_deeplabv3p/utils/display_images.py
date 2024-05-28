import torch
import pytorch_lightning as pl
from torchvision import transforms as transform_lib
import torchmetrics
from dataset.woodscape import WoodScapeDataset
from dataset.woodscape import WoodScapeDataModule
from dataset.cityscapes import Cityscapes
from dataset.cityscapes import CityscapesDataModule
from pl_bolts.callbacks import TrainingDataMonitor
from pl_bolts.callbacks import PrintTableMetricsCallback
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import make_grid
import torchvision.transforms.functional as tf


def show_image(img):
    img_np = img.cpu().numpy()
    img_reshaped = np.rollaxis(img_np, 0, 3)
    image = np.array(img_reshaped)  # image rgb (normalized)
    plt.imshow(image)
    plt.show()


def show_prediction_images(img, pred, decode_segmap, size=(128, 128), label=None):
    img_np = img.cpu().numpy()
    img_reshaped = np.rollaxis(img_np, 0, 3)
    image = np.array(img_reshaped)  # image rgb (normalized)

    pred_mask_np = pred.cpu().numpy()
    pred_mask_reshaped = pred_mask_np.reshape(size)
    pred_mask = np.array(pred_mask_reshaped)
    pred_mask_rgb = decode_segmap(pred_mask_np)  # pred mask rgb (color decoded)

    fig = plt.figure()
    if label is not None:
        label_np = label.cpu().numpy()
        label_reshaped = label_np.reshape(size)
        label_mask = np.array(label_reshaped)
        label_mask_rgb = decode_segmap(label_np)

        # add 1 x 3 plot
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

        ax1.title.set_text("Input Image")
        ax1.imshow(image)
        ax2.title.set_text("Label GT")
        ax2.imshow(label_mask_rgb)
        ax3.title.set_text("Prediction")
        ax3.imshow(pred_mask_rgb)
    else:
        # add 1 x 2 plot
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.title.set_text("Input Image")
        ax1.imshow(image)
        ax2.title.set_text("Prediction")
        ax2.imshow(pred_mask_rgb)

    plt.show()


def denormalize_img(img, norm_mean=None, norm_std=None):
    # Denormalize WoodScape Images for display
    # mean = [0.32757252, 0.33050337, 0.33689716],
    # std = [0.20290555, 0.20525302, 0.2037721]
    if norm_mean is None and norm_std is None:
        norm_mean = [0.32757252, 0.33050337, 0.33689716]
        norm_std = [0.20290555, 0.20525302, 0.2037721]

    inv_normalize = transform_lib.Normalize(
        mean=[-norm_mean[0]/norm_std[0], -norm_mean[1]/norm_std[1], -norm_mean[2]/norm_std[2]],
        std=[1 / norm_std[0], 1 / norm_std[1], 1 / norm_std[2]]
    )

    return inv_normalize(img)


def show_dataset_image(img, norm_mean=None, norm_std=None):
    img = img.detach()
    image = denormalize_img(img, norm_mean, norm_std)
    img_np = image.cpu().numpy()
    img_reshaped = np.rollaxis(img_np, 0, 3)
    image = np.array(img_reshaped)  # image rgb (normalized)
    plt.imshow(image)
    # plt.show()


def show_dataset_mask(img_mask, decode_segmap):
    img_mask = img_mask.detach()
    img_mask_np = img_mask.cpu().numpy()
    label_mask_rgb = decode_segmap(img_mask_np)
    # image = np.array(img_reshaped)  # image rgb (normalized)
    plt.imshow(label_mask_rgb)
    # plt.show()


def show_prediction_uncertainty_images(img, pred, uncertainty, decode_segmap, size=(128, 128), label=None):
    # img_normalized = torch.clone(img)
    image = denormalize_img(img)
    img_np = image.cpu().numpy()
    # img_np = img.cpu().numpy()
    img_reshaped = np.rollaxis(img_np, 0, 3)
    image = np.array(img_reshaped)  # image rgb (normalized)

    pred_mask_np = pred.cpu().numpy()
    pred_mask_reshaped = pred_mask_np.reshape(size)
    pred_mask = np.array(pred_mask_reshaped)
    pred_mask_rgb = decode_segmap(pred_mask_np)  # pred mask rgb (color decoded)

    uncertainty_np = uncertainty.cpu().numpy()
    uncertainty_np = np.squeeze(uncertainty_np)
    uncertainty_map = np.array(uncertainty_np)

    # fig = plt.figure()
    fig = plt.figure(figsize=(16,16), dpi= 100)
    if label is not None:
        label_np = label.cpu().numpy()
        label_reshaped = label_np.reshape(size)
        label_mask = np.array(label_reshaped)
        label_mask_rgb = decode_segmap(label_np)

        # add 1 x 4 plot
        ax1 = fig.add_subplot(411)
        ax2 = fig.add_subplot(412)
        ax3 = fig.add_subplot(413)
        ax4 = fig.add_subplot(414)

        ax1.title.set_text("Input Image")
        ax1.imshow(image)
        ax2.title.set_text("Label GT")
        ax2.imshow(label_mask_rgb)
        ax3.title.set_text("Semantic Map")
        ax3.imshow(pred_mask_rgb)
        ax4.title.set_text("Uncertainty Map (H)")
        ax4.imshow(uncertainty_map, cmap="inferno")
    else:
        # add 1 x 3 plot
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)

        ax1.title.set_text("Input Image")
        ax1.imshow(image)
        ax2.title.set_text("Semantic Map")
        ax2.imshow(pred_mask_rgb)
        ax3.title.set_text("Uncertainty Map (H)")
        ax3.imshow(uncertainty_map, cmap="inferno")

    plt.show()


def img2tensor(file):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    img = Image.open(file)
    img = img.resize((640, 483), Image.BILINEAR)
    img = tf.to_tensor(img)
    img = tf.normalize(img, norm_mean, norm_std, False)
    img = img.unsqueeze(0).to(device)
    return img

