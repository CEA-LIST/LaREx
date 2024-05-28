from typing import Tuple, List, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
import torchmetrics



class LeNet(nn.Module):
    """LeNet Neural Network Architecture
    Author: Andrei Bursuc
    https://github.com/abursuc/dldiy-gtsrb
    """
    def __init__(self, num_classes=43, input_channels=3):

        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear((16 * 5 * 5), 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)

        if num_classes == 1:
            # compatible with nn.BCELoss
            self.output = nn.Sigmoid()
        else:
            # compatible with nn.NLL loss
            self.output = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        
        out = F.relu(self.conv2(out))        
        out = F.max_pool2d(out, 2)
        
        out = out.view(out.size(0), -1)
        
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        out = self.output(out)
        return out


def lenet5(**kwargs):
    model = LeNet(**kwargs)
    return model


def lenet(model_name, num_classes, input_channels, pretrained=False):
    return{
        'lenet5': lenet5(num_classes=num_classes, input_channels=input_channels),
    }[model_name]


class LeNetModule(pl.LightningModule):
    """LeNet PyTorch-Lightning Module
    Author: Fabio Arnez

    """
    def __init__(self,
                 num_classes = 18,
                 loss_fn: str = 'nll',
                 optimizer_lr: float = 1e-4,
                 optimizer_weight_decay: float = 1e-5,
                 max_nro_epochs: int = None) -> None:
        super().__init__()
        
        if loss_fn not in ["nll","focal"]:
            raise ValueError(f' loss_fn value is not supported. Got "{loss_fn}" value.')
        
        self.num_classes = num_classes
        self.loss_fn = self.get_loss_fn(loss_fn)
        self.optimizer_lr = optimizer_lr
        self.optimizer_weight_decay = optimizer_weight_decay
        self.max_nro_epochs = max_nro_epochs
        # model: dataset has 18 classes, input is RGB  images (3 channels)
        self.model = LeNet(num_classes=18, input_channels=3) 
        # add Accuracy Metric
        self.metric_accuracy = torchmetrics.Accuracy(num_classes=self.num_classes)
        self.save_hyperparameters()  # save model hyperparameters!
        
    def get_loss_fn(self, loss_type) -> Any:
        if loss_type == "nll":
            loss_fn = nn.NLLLoss(reduction='mean')
        else:  # add focal loss
            # ToDo! Add the focal loss!
            pass
        return loss_fn
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(),  # specify your model (neural network) parameters (weights)
                                     lr=self.optimizer_lr,  # learning rate
                                     weight_decay=self.optimizer_weight_decay,  # L2 penalty regularizer
                                     eps=1e-7)  # adds numerical numerical stability (avoids division by 0)
        return [optimizer]
    
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
    