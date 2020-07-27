
import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import CRITERIA
from .sampler import build_sampler
# from .utils import weight_reduce_loss


@CRITERIA.register_module
class MultiRectBCELoss(nn.Module):

    def __init__(self,
                 sampler=None,
                 balanced=False,
                 weight=None,
                 reduction='mean',
                 ignore_index=255,
                 loss_weight=1.0,
                 label_epsilon=-1):
        super(MultiRectBCELoss, self).__init__()
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
                pred,
                label,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        mask_score, cls_score = pred
        mask_label, cls_label = label

        mask = (mask_label != self.ignore_index)

        if self.label_epsilon < 0:
            mask[:, 1:, :, :] &= mask_label[:, 0:1, :, :] == 1
        else:
            mask[:, 1:, :, :] &= mask_label[:, 0:1, :, :] == 1 - self.label_epsilon

        # import pdb
        # pdb.set_trace()

        loss_mask = self.loss_weight * F.binary_cross_entropy_with_logits(
            mask_score[mask], mask_label.float()[mask], reduction=reduction)

        loss_cls = 0.04 * F.binary_cross_entropy_with_logits(
            cls_score, cls_label.float(), reduction=reduction)

        return loss_mask + loss_cls
