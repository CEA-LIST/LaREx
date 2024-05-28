import torch.nn as nn
import torch.nn.functional as F
import torch

torch.autograd.set_detect_anomaly(True)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=0.0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class ELBO(nn.Module):
    def __init__(self, pred_loss_type="cross_entropy", img_pred_weight=2.0, beta_kl=4.0):
        super(ELBO, self).__init__()
        self.recon_loss_type = pred_loss_type
        self.img_pred_weight = img_pred_weight
        self.beta_kl = beta_kl
        self.img_pred_loss = 0.0
        self.loss_KLD = 0.0
        self.total_loss = 0.0

        if pred_loss_type == 'focal_loss':
            self.pred_loss = FocalLoss(size_average=True)
        elif pred_loss_type == 'cross_entropy':
            self.pred_loss = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
        else:
            raise ValueError("No pred_loss type available! Choose a valid type!")

    def forward(self, preds, targets, mu, logvar):
        self.img_pred_loss = self.pred_loss(preds, targets)
        self.loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        self.total_loss = self.img_pred_loss + (self.beta_kl * self.loss_KLD)
        return self.total_loss


class ELBOWeightVILoss(nn.Module):
    def __init__(self, pred_loss_type="cross-entropy", img_pred_weight=1.0):
        super(ELBOWeightVILoss, self).__init__()
        self.pred_loss_type = pred_loss_type
        self.img_pred_weight = img_pred_weight
        self.beta_kl = 1.0

        self.img_pred_loss = 0.0
        self.loss_KLD = 0.0
        self.total_loss = 0.0
        # self.train_size = train_size
        if self.pred_loss_type == 'focal_loss':
            self.pred_loss = FocalLoss(alpha=1.0, gamma=2.0, size_average=True)
        elif self.pred_loss_type == 'cross_entropy':
            self.pred_loss = nn.CrossEntropyLoss(reduction='mean')
        else:
            raise ValueError("No pred_loss type available! Choose a valid type!")

    def forward(self, preds, targets, kl, beta_kl):
        self.img_pred_loss = self.pred_loss(preds, targets)
        self.loss_KLD = kl
        self.beta_kl = beta_kl

        self.total_loss = (self.img_pred_weight * self.img_pred_loss) + (self.beta_kl * self.loss_KLD)
        return self.total_loss
        # assert not targets.requires_grad
        # return F.nll_loss(preds, targets, reduction='mean') * self.train_size + beta * kl


def calculate_kl(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl


def get_beta(batch_idx, m, beta_type):
    if type(beta_type) is float:
        return beta_type

    if beta_type == "Blundell":
        beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
    elif beta_type == "Standard":
        beta = 1 / m
    else:
        beta = 0
    return beta

# class WeightVIELBOLoss(nn.Module):
#     def __init__(self, train_size):
#         super(WeightVIELBOLoss, self).__init__()
#         self.train_size = train_size
#
#     def forward(self, pred, target, kl, beta):
#         assert not target.requires_grad
#         return F.nll_loss(pred, target, reduction='mean') * self.train_size + beta * kl
