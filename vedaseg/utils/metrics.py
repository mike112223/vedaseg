import numpy as np
import torch


def dice_score(pred, gt, thres_range=np.arange(0.1, 1.0, 0.1)):
    """dice_score
        
        Args:
            pred, n*c*h*w, torch.Tensor
            gt, n*c*h*w, torch.Tensor

        Return:
            dice, nthres * nclasses
    """
    gt = gt.float()

    dices = []
    for thres in thres_range:
        tpred = (pred > thres).float()
        nu = 2 * (tpred * gt).sum(dim=[2, 3])
        de = tpred.sum(dim=[2, 3]) + gt.sum(dim=[2, 3])
        dice = nu / de
        dice[torch.isnan(dice)] = 1
        dices.append(dice.sum(0))
    return torch.stack(dices, 0)


class MetricMeter(object):
    """MetricMeter
    """
    def __init__(self, nclasses):
        self.nclasses = nclasses
        self.confusion_matrix = np.zeros((self.nclasses,)*2)

    def pixel_accuracy(self):
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc

    def pixel_accuracy_class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def miou(self):
        IoUs = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix) + np.finfo(np.float32).eps)
        mIoU = np.nanmean(IoUs)
        return mIoU, IoUs

    def fw_iou(self):
        """fw_iou, frequency weighted iou
        """
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        IoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        fwIoU = (freq[freq > 0] * IoU[freq > 0]).sum()
        return fwIoU

    def _generate_matrix(self, pred, gt):
        mask = (gt >= 0) & (gt < self.nclasses)
        label = self.nclasses * gt[mask].astype('int') + pred[mask]
        count = np.bincount(label, minlength=self.nclasses**2)
        confusion_matrix = count.reshape(self.nclasses, self.nclasses)
        return confusion_matrix

    def add(self, pred, gt):
        assert pred.shape == gt.shape
        self.confusion_matrix += self._generate_matrix(pred, gt)

    def reset(self):
        self.confusion_matrix = np.zeros((self.nclasses,) * 2)


class MultiLabelMetricMeter(object):
    """MetricMeter
    """
    def __init__(self, nclasses, extra_super=False):
        self.nclasses = nclasses
        self.confusion_matrix = np.zeros((self.nclasses, 2))
        self.extra_super = extra_super

    def miou(self):
        IoUs = (self.confusion_matrix[:, 0] + 1e-9) / (self.confusion_matrix[:, 1] + 1e-9)
        mIoU = np.nanmean(IoUs)
        return mIoU, IoUs

    def _generate_matrix(self, pred, gt):
        b, c, h, w = gt.shape
        pred = pred.transpose(1, 0, 2, 3)
        gt = gt.transpose(1, 0, 2, 3)
        # print(pred.shape, gt.shape)

        confusion_matrix = []
        for i in range(c):
            sub_pred = pred[i]
            sub_gt = gt[i]

            mask = (sub_gt >= 0) & (sub_gt <= 1)
            sub_pred = sub_pred[mask]
            sub_gt = sub_gt[mask]

            sub_pred = 1 / (1 + np.exp(-sub_pred))

            if self.extra_super and i != 0:
                fg_pred = pred[0][mask]
                fg_pred = 1 / (1 + np.exp(-fg_pred))
                sub_pred *= fg_pred

            sub_pred = sub_pred > 0.5
            sub_gt = sub_gt == 1

            interactions = np.sum(sub_pred & sub_gt)
            unions = np.sum(sub_pred | sub_gt)

            confusion_matrix.append((interactions, unions))
        confusion_matrix = np.array(confusion_matrix)
        # print(confusion_matrix)

        return confusion_matrix

    def add(self, pred, gt):
        assert pred.shape == gt.shape
        self.confusion_matrix += self._generate_matrix(pred, gt)

    def reset(self):
        self.confusion_matrix = np.zeros((self.nclasses, 2))
