from typing import Optional, Any
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
import pytorch_lightning as pl
import torchmetrics
# from torchmetrics import Accuracy, JaccardIndex
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
# from deeplab import deeplab_v3plus
# from deeplab_v3p.deeplab import deeplab_v3plus
from deeplab_v3p import deeplab_v3plus
# from loss import FocalLoss, ELBO
from .loss import FocalLoss, ELBOWeightVILoss, get_beta
from torch.optim.lr_scheduler import CosineAnnealingLR
from .scheduler import PolyLR
import glob
from utils import img2tensor
from torchvision.utils import make_grid
from icecream import ic
torch.autograd.set_detect_anomaly(True)


class DeepLabV3PlusModule(pl.LightningModule):
    def __init__(self,
                 backbone_name='resnet101',
                 deeplabv3plus_type="normal",
                 dataset="woodscape",
                 n_class=10,
                 output_stride=16,
                 pretrained_backbone=True,
                 optimizer_lr=0.01,
                 optimizer_momentum=0.9,
                 optimizer_weight_decay=5e-4,
                 pred_loss_type="cross_entropy",
                 img_pred_weight=2.0,
                 len_train_loader_beta=1,
                 len_val_loader_beta=1,
                 max_nro_epochs=100,
                 test_images_path="./test_images/*.png",
                 label_colours=None) -> None:
        super().__init__()

        self.backbone_name = backbone_name
        self.deeplabv3plus_type = deeplabv3plus_type
        self.dataset = dataset
        self.n_class = n_class
        self.output_stride = output_stride
        self.pretrained_backbone = pretrained_backbone
        self.optimizer_lr = optimizer_lr
        self.optimizer_momentum = optimizer_momentum
        self.optimizer_weight_decay = optimizer_weight_decay
        self.pred_loss_type = pred_loss_type
        self.img_pred_weight = img_pred_weight
        self.len_train_loader_beta = len_train_loader_beta
        self.len_val_loader_beta = len_val_loader_beta
        self.max_nro_epochs = max_nro_epochs
        self.test_images_path = test_images_path
        self.label_colours = label_colours
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.deeplab_v3plus_model = deeplab_v3plus(self.backbone_name,
                                                   model_type=self.deeplabv3plus_type,
                                                   num_classes=self.n_class,
                                                   output_stride=self.output_stride,
                                                   pretrained_backbone=self.pretrained_backbone)
        # Get criterion:
        self.criterion = self.get_criterion()
        # Define metrics:
        self.metric_accuracy = torchmetrics.Accuracy(num_classes=self.n_class, mdmc_average='samplewise')
        self.metric_jaccard_idx = torchmetrics.JaccardIndex(num_classes=self.n_class)
        self.save_hyperparameters()

    def get_criterion(self):
        if self.deeplabv3plus_type == "normal" or self.deeplabv3plus_type == "backbone_dropblock2d":
            if self.pred_loss_type == 'focal_loss':
                criterion = FocalLoss(gamma=2.0, ignore_index=255, size_average=True)
            elif self.pred_loss_type == 'cross_entropy':
                criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
            else:
                raise ValueError("Not a valid loss type!")

        elif self.deeplabv3plus_type == "variational_layer":
            if self.pred_loss_type == 'focal_loss':
                criterion = ELBOWeightVILoss(pred_loss_type="focal_loss",
                                             img_pred_weight=self.img_pred_weight)
            elif self.pred_loss_type == 'cross_entropy':
                criterion = ELBOWeightVILoss(pred_loss_type="cross_entropy",
                                             img_pred_weight=self.img_pred_weight)
            else:
                raise ValueError("Not a valid loss type!")
        else:
            raise ValueError("DeepLabV3+ Model Type not available! Choose a valid type!")

        return criterion

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.deeplab_v3plus_model.parameters(),
                                    lr=self.optimizer_lr,
                                    momentum=self.optimizer_momentum,
                                    weight_decay=self.optimizer_weight_decay)

        # lr_scheduler = {"scheduler": PolyLR(optimizer,
        #                                     max_iters=self.max_nro_epochs,
        #                                     power=0.9),
        #                 "monitor": "validation_IoU"}
        lr_scheduler = {"scheduler": CosineAnnealingLR(optimizer,
                                                       T_max=self.max_nro_epochs,
                                                       eta_min=1e-3)}
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        img, mask = batch
        mask = mask.squeeze()
        if self.deeplabv3plus_type == "variational_layer":
            img_pred, kl = self.deeplab_v3plus_model.forward(img)
            beta_kl = get_beta(batch_idx, self.len_train_loader_beta, "Blundell")
            train_loss = self.criterion(img_pred, mask, kl, beta_kl)
            accuracy = self.metric_accuracy(img_pred, mask)
            jaccard_idx = self.metric_jaccard_idx(img_pred, mask)
            self.log_dict({"train_loss": train_loss,
                           "train_accuracy": accuracy,
                           "train_IoU": jaccard_idx}, on_step=False, on_epoch=True, prog_bar=True)

        else:  # deeplabv3+ normal or deeplabv3+ backbone_dropblock2d
            img_pred = self.deeplab_v3plus_model.forward(img)
            train_loss = self.criterion(img_pred, mask)
            # ToDo: the following inplace operation is causing errors:
            # mask[mask == 255] = 19  # cityscapes only!
            if self.dataset == "cityscapes":
                mask = torch.where(mask == 255, 19, mask)
            accuracy = self.metric_accuracy(img_pred, mask)
            jaccard_idx = self.metric_jaccard_idx(img_pred, mask)
            self.log_dict({"train_loss": train_loss,
                           "train_accuracy": accuracy,
                           "train_IoU": jaccard_idx}, on_step=False, on_epoch=True, prog_bar=True)

        return train_loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        img, mask = batch
        mask = mask.squeeze()
        if self.deeplabv3plus_type == "variational_layer":
            img_pred, kl = self.deeplab_v3plus_model.forward(img)
            beta_kl = get_beta(batch_idx, self.len_train_loader_beta, "Blundell")
            val_loss = self.criterion(img_pred, mask, kl, beta_kl)
            accuracy = self.metric_accuracy(img_pred, mask)
            jaccard_idx = self.metric_jaccard_idx(img_pred, mask)
            self.log_dict({"validation_loss": val_loss,
                           "validation_accuracy": accuracy,
                           "validation_IoU": jaccard_idx}, on_step=False, on_epoch=True, prog_bar=True)

        else:  # deeplabv3+ normal or deeplabv3+ backbone_dropblock2d
            img_pred = self.deeplab_v3plus_model.forward(img)
            val_loss = self.criterion(img_pred, mask)
            # ToDo: the following inplace operation is causing errors:
            # mask = torch.where(mask == 255, 19, mask)
            # mask[mask == 255] = 19  # cityscapes only!
            if self.dataset == "cityscapes":
                mask = torch.where(mask == 255, 19, mask)
            accuracy = self.metric_accuracy(img_pred, mask)
            jaccard_idx = self.metric_jaccard_idx(img_pred, mask)
            self.log_dict({"validation_loss": val_loss,
                           "validation_accuracy": accuracy,
                           "validation_IoU": jaccard_idx}, on_step=False, on_epoch=True, prog_bar=True)

        return val_loss

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        pass

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        pass

    def on_train_epoch_end(self) -> None:
        if self.label_colours is not None:
            ic("Epoch Ends: Test images!")
            img_folder = glob.glob(self.test_images_path)
            ic(img_folder)
            views = []
            results = []
            self.eval()
            with torch.no_grad():
                for i, file in enumerate(img_folder):
                    img = img2tensor(file)
                    if self.deeplabv3plus_type == "variational_layer":
                        pred_img, kl = self.deeplab_v3plus_model(img)
                    else:
                        pred_img = self.deeplab_v3plus_model(img)
                    pred = torch.argmax(pred_img, dim=1)
                    prediction = torch.FloatTensor([self.label_colours[p.item()] for p in pred.view(-1)]).to(self.device)
                    prediction = prediction.transpose(0, 1).view(3, 483, 640) / 255.
                    prediction = prediction.unsqueeze(0)
                    # overlay img and prediction:
                    img = (img - img.min()) / (img.max() - img.min())
                    both = 0.5 * img + 0.5 * prediction

                    res = torch.cat((img, prediction, both), dim=0)
                    grid = make_grid(res, nrow=4, padding=0)
                    results.append(grid.unsqueeze(0))
                    res = transforms.ToPILImage()(grid.squeeze_(0))
                    views.append(res)
                    ic(f"Test: {i / len(img_folder) * 100:.1f} %")

            results_t = torch.cat(results)
            # ic(results_t.shape)
            grid_log = make_grid(results_t, nrow=1, padding=0)
            self.logger.experiment.add_image("pred_test_images", grid_log, global_step=self.global_step)

            for i in range(len(views)):
                views[i].save(f'./test_results/img_{i:03d}.png')

        else:
            pass
