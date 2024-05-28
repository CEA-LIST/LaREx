from typing import Any, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
import torchmetrics
from torch.optim.lr_scheduler import CosineAnnealingLR
from .resnet_custom import resnet18
from loss import FocalLoss
from numpy import array


class ResnetModule(pl.LightningModule):
    """Resnet PyTorch-Lightning Module

    """
    def __init__(self,
                 arch_name: str = 'resnet18',
                 input_channels: int = 3,
                 num_classes: int = 43,
                 spectral_norm: bool = False,
                 dropblock: bool = False,
                 dropblock_prob: float = 0.0,
                 dropblock_location: int = 2,
                 dropblock_block_size: int = 3,
                 dropout: bool = False,
                 dropout_prob: float = 0.0,
                 activation: str = "relu",
                 avg_pool: bool = False,
                 loss_fn: str = 'cross_entropy',
                 optimizer_lr: float = 1e-4,
                 optimizer_weight_decay: float = 1e-5,
                 max_nro_epochs: int = None,
                 ash: bool = False,
                 ash_percentile: int = 80,
                 dice_precompute: bool = False,
                 dice_inference: bool = False,
                 dice_p: int = 90,
                 dice_info: Union[None, array] = None,
                 react_threshold: Union[None, float] = None,
                 spectral_norm_only_fc: bool = False,
                 batch_norm: bool = True) -> None:
        super().__init__()
        
        if arch_name not in ["resnet18","resnet34", "resnet50", "resnet101", "resnet152"]:
            raise ValueError(f'arch_name value is not supported. Got "{arch_name}" value.')
        
        if loss_fn not in ["nll", "cross_entropy", "focal"]:
            raise ValueError(f'loss_fn value is not supported. Got "{loss_fn}" value.')
                
        self.arch_name = arch_name
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.spectral_norm = spectral_norm
        self.dropblock = dropblock
        self.dropblock_prob = dropblock_prob
        self.dropblock_block_size = dropblock_block_size
        self.dropout = dropout
        self.dropout_prob = dropout_prob
        self.loss_fn = self.get_loss_fn(loss_fn)
        self.optimizer_lr = optimizer_lr
        self.optimizer_weight_decay = optimizer_weight_decay
        self.max_nro_epochs = max_nro_epochs
        # ToDo: add different types of models
        # model: dataset has 43 classes, input is RGB  images (3 channels)
        self.model = resnet18(input_channels=self.input_channels,
                              num_classes=self.num_classes,
                              dropblock=self.dropblock,
                              dropblock_prob=self.dropblock_prob,
                              dropblock_location=dropblock_location,
                              dropblock_block_size=self.dropblock_block_size,
                              dropout=self.dropout,
                              dropout_prob=self.dropout_prob,
                              spectral_norm=self.spectral_norm,
                              activation=activation,
                              avg_pool=avg_pool,
                              ash=ash,
                              ash_percentile=ash_percentile,
                              dice_precompute=dice_precompute,
                              dice_inference=dice_inference,
                              dice_p=dice_p,
                              dice_info=dice_info,
                              react_threshold=react_threshold,
                              spectral_norm_only_fc=spectral_norm_only_fc,
                              batch_norm=batch_norm)
        # add Accuracy Metric
        self.metric_accuracy = torchmetrics.Accuracy(num_classes=self.num_classes)
        
        # save model hyperparameters:
        self.save_hyperparameters()
        
    def get_loss_fn(self, loss_type) -> Any:
        if loss_type == "nll":
            loss_fn = nn.NLLLoss(reduction='mean')
        elif loss_type == "cross_entropy":
            loss_fn = nn.CrossEntropyLoss(reduction='mean')
        elif loss_type == "focal":
            loss_fn = FocalLoss(gamma=4.0, reduction='mean')
        else:
            raise ValueError(f' Loss function value is not supported, use a valid loss function')

        return loss_fn
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(),  # specify your model (neural network) parameters (weights)
                                     lr=self.optimizer_lr,  # learning rate
                                     weight_decay=self.optimizer_weight_decay,  # L2 penalty regularizer
                                     eps=1e-7)  # adds numerical numerical stability (avoids division by 0)
        lr_scheduler = {"scheduler": CosineAnnealingLR(optimizer,
                                                       T_max=self.max_nro_epochs,
                                                       eta_min=1e-5)}
        return [optimizer], [lr_scheduler]
    
    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        sample, label = batch

        pred = self.model(sample)  # make a prediction (forward-pass)

        train_loss = self.loss_fn(pred, label)  # compute loss

        train_accuracy = self.metric_accuracy(pred, label)  # compute accuracy
        
        self.log_dict({"Train loss": train_loss,
                       "Train accuracy": train_accuracy},
                      on_step=False, on_epoch=True, prog_bar=True)

        return train_loss
    
    def validation_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        sample, label = batch
        
        pred = self.model(sample)
        
        valid_loss = self.loss_fn(pred, label)
        
        valid_accuracy = self.metric_accuracy(pred, label)
        
        self.log_dict({"Validation loss": valid_loss,
                       "Validation accuracy": valid_accuracy},
                      on_step=False, on_epoch=True, prog_bar=True)        

        return valid_loss
    
    def test_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        # ToDo: if required
        pass
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        # ToDo: if required
        pass