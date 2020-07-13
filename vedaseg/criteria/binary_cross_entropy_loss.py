
import torch.nn as nn
import torch.nn.functional as F

from .registry import CRITERIA
from .sampler import build_sampler
# from .utils import weight_reduce_loss


@CRITERIA.register_module
class BCELoss(nn.Module):

    def __init__(self,
                 sampler=None,
                 balanced=False,
                 weight=None,
                 reduction='mean',
                 ignore_index=255,
                 loss_weight=1.0):
        super(BCELoss, self).__init__()
        self.balanced = balanced
        self.weight = weight
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight

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

        loss_cls = self.loss_weight * F.binary_cross_entropy_with_logits(
            cls_score[mask], label.float()[mask], weight=weight,
            reduction=reduction)

        return loss_cls
