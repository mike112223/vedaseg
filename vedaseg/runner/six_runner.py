
import sys
import math
import logging
import os.path as osp

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from collections.abc import Iterable
from tqdm import tqdm
import matplotlib.pyplot as plt

from vedaseg.utils.checkpoint import load_checkpoint, save_checkpoint
from .registry import RUNNERS

np.set_printoptions(precision=4)

logger = logging.getLogger()


@RUNNERS.register_module
class SIXRunner(object):
    """ Runner

    """
    def __init__(self,
                 loader,
                 model,
                 criterion,
                 metric,
                 optim,
                 lr_scheduler,
                 max_epochs,
                 workdir,
                 start_epoch=0,
                 trainval_ratio=1,
                 snapshot_interval=1,
                 gpu=True,
                 test_cfg=None,
                 test_mode=False,
                 save_fpfn=False,
                 show_fpfn=False):
        self.loader = loader
        self.model = model
        self.criterion = criterion
        self.metric = metric
        self.optim = optim
        self.lr_scheduler = lr_scheduler
        self.start_epoch = start_epoch
        self.max_epochs = max_epochs
        self.workdir = workdir
        self.trainval_ratio = trainval_ratio
        self.snapshot_interval = snapshot_interval
        self.gpu = gpu
        self.test_cfg = test_cfg
        self.test_mode = test_mode
        self.save_fpfn = save_fpfn
        self.show_fpfn = show_fpfn

        self.best_fg_iou = 0

        self.ncls = 5

    def __call__(self, ap_ana=False,
                 conf_thresholds=None, size_thresholds=None):
        if ap_ana:

            tps, tns, fps, fns, p, r, tpr, fpr = self.judge_epoch(
                conf_thresholds=conf_thresholds,
                size_thresholds=size_thresholds
            )

            with np.printoptions(precision=4, suppress=True):
                for i in range(self.ncls):
                    print('======================')
                    print('precision, recall, fpr')
                    print(p[:, :, i], r[:, :, i], fpr[:, :, i])

    def judge_epoch(self, conf_thresholds=None, size_thresholds=None):
        c_l, i_l, s_l, cl_l = len(conf_thresholds), \
                              len(size_thresholds), \
                              len(self.loader['val']), \
                              self.ncls

        precision = np.zeros(shape=(c_l, i_l, cl_l))
        recall = np.zeros_like(precision)
        tp_rates = np.zeros_like(precision)
        fp_rates = np.zeros_like(precision)
        tps = np.zeros_like(precision)
        tns = np.zeros_like(precision)
        fps = np.zeros_like(precision)
        fns = np.zeros_like(precision)

        total_res = np.zeros(shape=(c_l, i_l, s_l, cl_l, 4))
        gt = np.zeros(len(self.loader['val']))
        for sample_id, (img, label, ori_img, file_name) in enumerate(
                tqdm(self.loader['val'],
                     desc=f'Inference with different thresholds',
                     dynamic_ncols=True)
        ):
            # print(file_name)

            # if sample_id not in (2, 3, 9, 92, 105, 111, 149, 155, 171, 183, 192, 197, 284, 298, 310, 328, 349, 386, 401):
            #     continue

            if img is None:
                continue
            else:
                shape = img.shape[-2:]
                h, w = shape[0], shape[1]
                if h * w >= 10000000:
                    continue

            res = self.judge_batch(img,
                                   label,
                                   conf_thresholds=conf_thresholds,
                                   size_thresholds=size_thresholds,
                                   ori_img=ori_img,
                                   sample_id=sample_id)
            total_res[:, :, sample_id, :, :] = res

        total_res = total_res.sum(axis=2)

        for conf_idx, conf_thres in enumerate(conf_thresholds):
            for size_idx, size_thres in enumerate(size_thresholds):
                # tp, tn, fp, fn = self.ana_tpfn(gt,
                #                                total_res[conf_idx, size_idx, :])
                for c in range(self.ncls):
                    tp, fp, tn, fn = total_res[conf_idx, size_idx, c]
                    p, r = self.ana_pr(tp, tn, fp, fn)
                    tpr, fpr = self.ana_roc(tp, tn, fp, fn)
                    precision[conf_idx, size_idx, c] = p
                    recall[conf_idx, size_idx, c] = r
                    tp_rates[conf_idx, size_idx, c] = tpr
                    fp_rates[conf_idx, size_idx, c] = fpr
                    tps[conf_idx, size_idx, c] = tp
                    tns[conf_idx, size_idx, c] = tn
                    fps[conf_idx, size_idx, c] = fp
                    fns[conf_idx, size_idx, c] = fn
        return tps, tns, fps, fns, precision, recall, tp_rates, fp_rates

    def judge_batch(self,
                    img, label,
                    conf_thresholds=None, size_thresholds=None, ori_img=None, sample_id=None):

        res = np.zeros((len(conf_thresholds), len(size_thresholds), self.ncls, 4))

        self.model.eval()
        with torch.no_grad():
            if self.gpu:
                img = img.cuda()
            label = label

            pred = self.model(img)

            # prob = pred.softmax(dim=1)

        for idx_c, conf_th in enumerate(conf_thresholds):
            for idx_i, size_th in enumerate(size_thresholds):
                res[idx_c, idx_i] = self.judge_conf_map(img, pred, label,
                                                        conf_thres=conf_th,
                                                        size_thres=size_th,
                                                        ori_img=ori_img,
                                                        sample_id=sample_id)
        return res

    def judge_conf_map(self, img, pred, label, conf_thres=None, size_thres=100, ori_img=None, sample_id=None):
        # tp fp tn fn
        label = [_.tolist()[0] for _ in label]
        tfpn = np.zeros((self.ncls, 4))

        prob = pred.sigmoid()[0].cpu().numpy()
        prob[1:, :, :] *= prob[0, :, :]

        if conf_thres is None:
            conf_thres = 0.5

        h, w = ori_img.shape[1:3]
        prob = prob[:, :h, :w]
        result = []

        for i in range(self.ncls):
            mask = cv2.threshold(
                prob[i], conf_thres, 1, cv2.THRESH_BINARY)[1]

            _contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in _contours:
                if cv2.contourArea(c) > size_thres:
                    result.append(1)
                    break
            else:
                result.append(0)

            if label[i] == 0 and result[i] == 0:
                tfpn[i][2] += 1
            elif label[i] == 1 and result[i] == 1:
                tfpn[i][0] += 1
            elif label[i] == 1 and result[i] == 0:
                tfpn[i][3] += 1
            else:
                tfpn[i][1] += 1

            if self.show_fpfn or self.save_fpfn:
                fp, fn = tfpn[i][1], tfpn[i][3]
                self._show_fpfn(ori_img, i, prob[i], label[i], conf_thres, size_thres, fp=fp, fn=fn, sample_id=sample_id)

        return tfpn

    @staticmethod
    def ana_pr(tp, tn, fp, fn):
        precision = tp / (tp + fp + sys.float_info.min)
        recall = tp / (tp + fn + sys.float_info.min)
        return precision, recall

    @staticmethod
    def ana_roc(tp, tn, fp, fn):
        tp_rate = tp / (tp + fn + sys.float_info.min)
        fp_rate = fp / (fp + tn + sys.float_info.min)
        return tp_rate, fp_rate

    def _show_fpfn(self, img, clsn, pred, label, conf_thres, size_thres, fp=None, fn=None, sample_id=None):
        # label[label == 255] = 0

        # img = img[0][:, :, (2, 1, 0)].numpy()
        # img = self.draw_mask(img, pred.cpu().numpy()[0])
        # img = self.draw_mask(img, (label.cpu().numpy()[0] == 1), c=(255, 0, 0))

        if fp or fn:

            # import pdb
            # pdb.set_trace()

            plt.figure(num=sample_id, figsize=(25, 8))
            plt.suptitle('cls %d & conf %.2f & size %.2f & %s' % (clsn, conf_thres, size_thres, 'fp' if fp else 'fn'))
            plt.subplot(1, 2, 1)
            plt.title('img')
            plt.imshow(img[0][:, :, (2, 1, 0)].int())
            plt.subplot(1, 2, 2)
            plt.title('pred')
            plt.imshow((255 * pred).astype(np.uint8))

            if self.save_fpfn:
                plt.savefig(osp.join(self.workdir, '%d_%d.png' % (sample_id, clsn)))

            if self.show_fpfn:
                plt.show()

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None):

        _, ious = self.metric.miou()

        best_flag = 0
        if np.mean(ious[1:]) > self.best_fg_iou:
            self.best_fg_iou = np.mean(ious[1:])
            best_flag = 1

        if self.epoch % self.snapshot_interval == 0 or self.epoch == self.max_epochs:
            if meta is None:
                meta = dict(epoch=self.epoch, iter=self.iter, lr=self.lr)
            else:
                meta.update(epoch=self.epoch, iter=self.iter, lr=self.lr)

            filename = filename_tmpl.format(self.epoch)
            filepath = osp.join(out_dir, filename)
            optimizer = self.optim if save_optimizer else None
            logger.info('Save checkpoint %s', filename)
            save_checkpoint(self.model,
                            filepath,
                            optimizer=optimizer,
                            meta=meta)

        if best_flag:
            if meta is None:
                meta = dict(epoch=self.epoch, iter=self.iter, lr=self.lr)
            else:
                meta.update(epoch=self.epoch, iter=self.iter, lr=self.lr)

            filename = filename_tmpl.format('best')
            filepath = osp.join(out_dir, filename)
            optimizer = self.optim if save_optimizer else None
            logger.info('Save checkpoint %s', filename)
            save_checkpoint(self.model,
                            filepath,
                            optimizer=optimizer,
                            meta=meta)

    def load_checkpoint(self, filename, map_location='cpu', strict=False):
        logger.info('Resume from %s', filename)
        return load_checkpoint(self.model, filename, map_location, strict,
                               logger)

    @property
    def epoch(self):
        """int: Current epoch."""
        return self.lr_scheduler.last_epoch

    @epoch.setter
    def epoch(self, val):
        """int: Current epoch."""
        self.lr_scheduler.last_epoch = val

    @property
    def lr(self):
        lr = [x['lr'] for x in self.optim.param_groups]
        return np.array(lr)

    @lr.setter
    def lr(self, val):
        for idx, param in enumerate(self.optim.param_groups):
            if isinstance(val, Iterable):
                param['lr'] = val[idx]
            else:
                param['lr'] = val

    @property
    def iter(self):
        """int: Current iteration."""
        return self.lr_scheduler.last_iter

    @iter.setter
    def iter(self, val):
        """int: Current epoch."""
        self.lr_scheduler.last_iter = val

    def resume(self,
               checkpoint,
               resume_optimizer=False,
               resume_lr=True,
               resume_epoch=True,
               map_location='default'):
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            checkpoint = self.load_checkpoint(
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = self.load_checkpoint(checkpoint, map_location=map_location)
        if 'optimizer' in checkpoint and resume_optimizer:
            self.optim.load_state_dict(checkpoint['optimizer'])
        if resume_epoch:
            self.epoch = checkpoint['meta']['epoch']
            self.start_epoch = self.epoch
            self.iter = checkpoint['meta']['iter']
        if resume_lr:
            self.lr = checkpoint['meta']['lr']
