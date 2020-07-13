
import torch
import torch.nn.functional as F

from .registry import SAMPLERS


@SAMPLERS.register_module
class OHEM(object):

    def __init__(self,
                 mode='img',
                 ratio=10000,
                 neg_num_bound=10000000,
                 ignore_index=255):
        super(OHEM, self).__init__()
        self.mode = mode
        self.ratio = ratio
        self.neg_num_bound = neg_num_bound
        self.ignore_index = ignore_index

        assert self.mode in ('batch', 'img')

    def sample(self,
               cls_score,
               label):
        preds = F.softmax(cls_score, dim=1)[:, 1, :, :]
        masks = label != 255

        if self.mode == 'img':
            sample_masks = self.ohem_batch(preds, label, masks)
        else:
            sample_masks = self.ohem(preds, label, masks)

        return sample_masks

    def ohem_single(self, pred, gt, mask):  # shape h*w
        pos_num = torch.sum(gt > 0.5) - torch.sum((gt > 0.5) & (mask <= 0.5))
        neg_num = torch.sum(gt <= 0.5)

        if pos_num == 0:
            # selected_mask = mask
            # return torch.unsqueeze(selected_mask, dim=0).float()
            neg_num = min(neg_num, self.neg_num_bound)
        else:
            neg_num = min(pos_num * 3, neg_num)

        neg_score = pred[gt <= 0.5]
        neg_score_sorted = torch.sort(-neg_score)[0]
        threshold = -neg_score_sorted[neg_num - 1]

        selected_mask = ((pred >= threshold) | (gt > 0.5)) & (mask > 0.5)
        return torch.unsqueeze(selected_mask, dim=0).float()

    def ohem_batch(self, preds, gts, masks):  # shape: B*H*W
        sample_masks = []
        for i in range(preds.shape[0]):
            sample_masks.append(
                self.ohem_single(preds[i, :, :], gts[i, :, :], masks[i, :, :]))

        sample_masks = torch.cat(sample_masks, 0)

        return sample_masks

    def ohem(self, preds, gts, masks):
        pos_num = torch.sum(gts == 1)
        neg_num = torch.sum(gts == 0)

        if pos_num == 0:
            # selected_mask = mask
            # return torch.unsqueeze(selected_mask, dim=0).float()
            neg_num = min(neg_num, self.neg_num_bound)
        else:
            neg_num = min(pos_num * self.ratio, neg_num)

        neg_score = preds[gts == 0]
        neg_score_sorted = torch.sort(-neg_score)[0]
        threshold = -neg_score_sorted[neg_num - 1]

        print(threshold)

        selected_mask = ((preds >= threshold) | (gts == 1)) & masks

        import cv2
        import numpy as np
        for i in range(len(selected_mask)):
            cv2.imwrite('workdir/debug/mask_%d.png'%i, 255 * selected_mask[i].cpu().numpy().astype(np.uint8))
        # import pdb
        # pdb.set_trace()

        return selected_mask.float()
