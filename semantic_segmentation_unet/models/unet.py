
from typing import Optional, Any
import torch
from .unet_blocks import *
from pytorch_lightning.utilities.types import STEP_OUTPUT
import pytorch_lightning as pl
import torchmetrics
from dropblock import DropBlock2D
from torch.optim.lr_scheduler import CosineAnnealingLR
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torch.optim.lr_scheduler import StepLR


class UnetSemSeg(nn.Module):
    """
    A UNet (https://arxiv.org/abs/1505.04597) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: list with the amount of filters per layer
    apply_last_layer: boolean to apply last layer or not
    padding: Boolean, if true we pad the images with 1 so that we keep the same dimensions
    """
    def __init__(self,
                 input_channels,
                 num_classes,
                 num_filters,
                 initializers,
                 drop_block2d=True,
                 apply_last_layer=True,
                 padding=True):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.drop_block2d = drop_block2d,
        self.padding = padding
        self.activation_maps = []
        self.apply_last_layer = apply_last_layer
        self.contracting_path = nn.ModuleList()

        for i in range(len(self.num_filters)):
            input = self.input_channels if i == 0 else output
            output = self.num_filters[i]

            if i == 0:
                pool = False
            else:
                pool = True

            self.contracting_path.append(DownConvBlock(input, output, initializers, padding, pool=pool))

        if self.drop_block2d:
            self.drop_block2d_layer = DropBlock2D(block_size=8, drop_prob=0.5)  # DropBlock2D
        
        self.upsampling_path = nn.ModuleList()

        n = len(self.num_filters) - 2
        for i in range(n, -1, -1):
            input = output + self.num_filters[i]
            output = self.num_filters[i]
            self.upsampling_path.append(UpConvBlock(input, output, initializers, padding))

        if self.apply_last_layer:
            self.last_layer = nn.Conv2d(output, self.num_classes, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.contracting_path):
            x = down(x)
            if i != len(self.contracting_path)-1:
                blocks.append(x)
        # ic(x.shape)
        if self.drop_block2d:
            x = self.drop_block2d_layer(x)
            # ic("encoder: ", x.shape)
        
        for i, up in enumerate(self.upsampling_path):
            x = up(x, blocks[-i - 1])
            # x = up(x, blocks[-i])

        del blocks
        
        if self.apply_last_layer:
            x = self.last_layer(x)
        return x


class UnetSemSegModule(pl.LightningModule):
    def __init__(self,
                 input_channels=3,
                 num_classes=20,
                 num_filters=None,
                 drop_block2d=True,
                 lr=1e-4,
                 pred_loss_type="cross_entropy",
                 max_nro_epochs=100) -> None:
        super().__init__()
        if num_filters is None:
            num_filters = [32, 64, 128, 192]
        self.num_filters = num_filters
        self.drop_block2d = drop_block2d
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.beta_annealing_lambda = None
        self.pred_loss_type = pred_loss_type
        self.max_nro_epochs = max_nro_epochs
        self.initializers = {'w': 'he_normal', 'b': 'normal'}

        self.unet_model = UnetSemSeg(input_channels=self.input_channels,
                                     num_classes=self.num_classes,
                                     num_filters=self.num_filters,
                                     initializers=self.initializers,
                                     drop_block2d=self.drop_block2d)
        self.lr = lr
        self.valid_batches_means = []
        self.loader_batches_means = []
        # get loss_fn
        self.loss_fn = self.get_criterion()
        # get metrics
        self.metric_accuracy = torchmetrics.Accuracy(num_classes=self.num_classes,
                                                     mdmc_average='samplewise',
                                                     ignore_index=None)
        self.metric_jaccard_idx = torchmetrics.JaccardIndex(num_classes=self.num_classes, ignore_index=None)
        self.save_hyperparameters()

    def get_criterion(self):
        if self.pred_loss_type == 'cross_entropy':
            criterion = nn.CrossEntropyLoss(reduction='mean')
        else:  # ToDo: Add other loss functions! e.g., Focal Loss
            raise ValueError("Not a valid loss type!")
        return criterion

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.unet_model.parameters(),
                                     lr=self.lr,
                                     weight_decay=0.0)
        # LR Scheduler: 2.5e-5 to 2.3e-5
        lr_scheduler = {"scheduler": CosineAnnealingLR(optimizer,
                                                       T_max=self.max_nro_epochs,
                                                       eta_min=2.3e-5)}
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        img, mask = batch
        img = img.squeeze()
        mask = mask.squeeze()
        pred_seg = self.unet_model(img)
        train_loss = self.loss_fn(pred_seg, mask)

        accuracy = self.metric_accuracy(pred_seg, mask)
        jaccard_idx = self.metric_jaccard_idx(pred_seg, mask)

        self.log_dict({"train_loss": train_loss,
                       "train_accuracy": accuracy,
                       "train_IoU": jaccard_idx}, on_step=False, on_epoch=True, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        img, mask = batch
        img = img.squeeze()
        mask = mask.squeeze()
        pred_seg = self.unet_model(img)
        val_loss = self.loss_fn(pred_seg, mask)
        accuracy = self.metric_accuracy(pred_seg, mask)
        jaccard_idx = self.metric_jaccard_idx(pred_seg, mask)

        self.log_dict({"validation_loss": val_loss,
                       "validation_accuracy": accuracy,
                       "validation_IoU": jaccard_idx}, on_step=False, on_epoch=True, prog_bar=True)

        return val_loss

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        pass

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        pass


if __name__ == "__main__":
    # num_filters = [32, 64, 128]
    num_filters = [32, 64, 128, 256]
    initializers = {'w': 'he_normal', 'b': 'normal'}
    unet = UnetSemSeg(3, 10, num_filters=num_filters, initializers=initializers, drop_block2d=True)
    ic(unet)
    x = torch.randn(1, 3, 128, 256)
    c = unet(x)
    ic(c.shape)
