
import torch.nn as nn
import torch.nn.functional as F

from .registry import CRITERIA
from .sampler import build_sampler
# from .utils import weight_reduce_loss


@CRITERIA.register_module
class CrossEntropyLoss(nn.Module):

    def __init__(self,
                 sampler=None,
                 balanced=False,
                 weight=None,
                 reduction='mean',
                 ignore_index=255,
                 loss_weight=1.0):
        super(CrossEntropyLoss, self).__init__()
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

        if self.sampler is not None:
            sample_mask = self.sampler.sample(cls_score.detach(), label.detach())
            label[sample_mask == 0] = self.ignore_index

        print('pos num {} / {} / {}'.format((label==1).sum(), (label==0).sum(), (label!=self.ignore_index).sum()))

        # loss_cls = self.cross_entropy(
        #     cls_score, label, sample_mask,
        #     weight=weight, reduction=reduction, avg_factor=sample_mask.sum())

        loss_cls = self.loss_weight * F.cross_entropy(
            cls_score, label, weight=weight,
            reduction=reduction, ignore_index=self.ignore_index)

        return loss_cls

    # def cross_entropy(self, pred, label, mask,
    #                   weight=None, reduction='mean', avg_factor=None):

    #     # element-wise losses
    #     loss = F.cross_entropy(
    #         pred, label, weight=weight, reduction='none')

    #     import pdb
    #     pdb.set_trace()

    #     loss *= mask.float()

    #     loss = weight_reduce_loss(
    #         loss, weight=None, reduction=reduction, avg_factor=avg_factor)

    #     return loss
