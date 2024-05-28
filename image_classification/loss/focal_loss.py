import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor

torch.autograd.set_detect_anomaly(True)


class FocalLoss(nn.Module):
    def __init__(self,
                 alpha: float = 1.0,
                 gamma: float = 0.0,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        
        super(FocalLoss, self).__init__()
        if reduction not in ['mean', 'none', 'sum']:
            raise NotImplementedError('Reduction {} not implemented.'.format(reduction))
        
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs,
                                  targets,
                                  reduction='none',
                                  ignore_index=self.ignore_index)
        
        pt = torch.exp(-ce_loss)
        
        focal_loss = self.alpha * ((1 - pt)**self.gamma) * ce_loss
        
        if self.reduction is 'mean':
            return focal_loss.mean()
        
        elif self.reduction is 'sum':
            return focal_loss.sum()
        
        else:
            return focal_loss
