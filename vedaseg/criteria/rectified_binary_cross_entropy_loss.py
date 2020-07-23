
import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import CRITERIA
from .sampler import build_sampler
# from .utils import weight_reduce_loss


@CRITERIA.register_module
class RectBCELoss(nn.Module):

    def __init__(self,
                 sampler=None,
                 balanced=False,
                 weight=None,
                 reduction='mean',
                 ignore_index=255,
                 loss_weight=1.0,
                 label_epsilon=-1):
        super(RectBCELoss, self).__init__()
        self.balanced = balanced
        self.weight = weight
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight
        self.label_epsilon = label_epsilon

        if sampler is not None:
            self.sampler = build_sampler(sampler)
        else:
            self.sampler = None

    def forward(self,
                cls_score,
                label,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.weight:
            weight = cls_score.new_tensor(self.weight)
        elif self.balanced:
            weight = cls_score.new_tensor([0, 0])
            weight[0] = (label == 1).sum()
            weight[1] = (label == 0).sum()
            weight /= weight.sum()
            if 1 in (weight == 0).tolist():
                weight = None
        else:
            weight = None

        mask = (label != self.ignore_index)

        if self.label_epsilon < 0:
            mask[:, 1:, :, :] &= label[:, 0:1, :, :] == 1
        else:
            mask[:, 1:, :, :] &= label[:, 0:1, :, :] == 1 - self.label_epsilon

        loss_cls = self.loss_weight * F.binary_cross_entropy_with_logits(
            cls_score[mask], label.float()[mask], weight=weight,
            reduction=reduction)

        return loss_cls
