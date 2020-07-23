
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
class MultiRunner(object):
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

        self.multilabel = 'multilabel' in self.metric.__class__.__name__.lower()
        self.ncls = len(self.loader['val'].dataset.cat_ids) + 1

    def __call__(self, ap_ana=False,
                 conf_thresholds=None, iou_thresholds=None):
        if ap_ana:

            tps, tns, fps, fns, p, r, tpr, fpr = self.judge_epoch(
                conf_thresholds=conf_thresholds,
                iou_thresholds=iou_thresholds
            )

            with np.printoptions(precision=4, suppress=True):
                for i in range(self.ncls):
                    print('======================')
                    print('precision, recall, fpr')
                    print(p[:, :, i], r[:, :, i], fpr[:, :, i])

        elif self.test_mode:
            self.test_epoch()
        else:
            assert self.trainval_ratio > 0
            for epoch in range(self.start_epoch, self.max_epochs):
                self.train_epoch()
                if self.trainval_ratio > 0 \
                        and (epoch + 1) % self.trainval_ratio == 0 \
                        and self.loader.get('val'):
                    self.validate_epoch()
                    self.save_checkpoint(self.workdir)

    def train_epoch(self):
        logger.info('Epoch %d, Start training' % self.epoch)
        iter_based = hasattr(self.lr_scheduler, '_iter_based')
        self.metric.reset()
        for img, label in self.loader['train']:
            self.train_batch(img, label)
            if iter_based:
                self.lr_scheduler.step()
        if not iter_based:
            self.lr_scheduler.step()

    def validate_epoch(self):
        logger.info('Epoch %d, Start validating' % self.epoch)
        self.metric.reset()
        for img, label in self.loader['val']:
            self.validate_batch(img, label)

    def test_epoch(self):
        logger.info('Start testing')
        logger.info('test info: %s' % self.test_cfg)
        self.metric.reset()
        for img, label, info, path in self.loader['val']:
            self.test_batch(img, label)

    def train_batch(self, img, label):
        self.model.train()

        self.optim.zero_grad()

        # for i in range(len(img)):
        #     mean = (123.675, 116.280, 103.530)
        #     std = (58.395, 57.120, 57.375)
        #     mean = np.reshape(np.array(mean, dtype=np.float32), [1, 1, 3])
        #     std = np.reshape(np.array(std, dtype=np.float32), [1, 1, 3])
        #     denominator = np.reciprocal(std, dtype=np.float32)
        #     cv2.imwrite('workdir/debug/img_%d_%d.png'%(self.iter, i), (img[i].cpu().numpy().transpose(1, 2, 0)/ denominator + mean).astype(np.uint8))
        #     for j in range(len(label[i])):
        #         cv2.imwrite('workdir/debug/label_%d_%d_%d.png' % (self.iter, i, j), label[i, j].numpy().astype(np.uint8) * 255)

        # import pdb
        # pdb.set_trace()

        if self.gpu:
            img = img.cuda()
            label = label.cuda()
        pred = self.model(img)
        loss = self.criterion(pred, label)
        mask_pred, cls_pred = pred

        loss.backward()
        self.optim.step()

        import pdb
        pdb.set_trace()

        with torch.no_grad():

            '''
            import matplotlib.pyplot as plt
            mask_pred = (prob[0]).permute(1, 2, 0).float().cpu().numpy()[:, :, 0]
            im = img[0].permute(1, 2, 0).clamp(min=0, max=1).cpu().numpy()
            label_ = label[0].permute(1, 2, 0).clamp(min=0, max=1).cpu().numpy()[:, :, 0]
            import random
            random_num = random.randint(0, 1000)
            mask_pred_name = 'output/%d_mask_pred.jpg' % random_num
            plt.imsave(mask_pred_name, mask_pred, cmap='Greys')
            im_name = 'output/%d.jpg' % random_num
            plt.imsave(im_name, im, cmap='Greys')
            label_name = 'output/%d_gt.jpg' % random_num
            plt.imsave(label_name, label_, cmap='Greys')
            '''
            if not self.multilabel:
                print('!!!!!!!!!!!')
                _, mask_pred = torch.max(mask_pred, dim=1)
            else:
                mask_pred = mask_pred.sigmoid()

            # if self.iter % 20 == 0:
            #     for i in range(4):
            #         cv2.imwrite(
            #             osp.join(self.workdir, 'mask_pred%d_%d.png' % (i, self.iter)),
            #             255 * mask_pred_label.cpu().numpy()[i].astype(np.uint8))

            self.metric.add(mask_pred.cpu().numpy(), label.cpu().numpy())
            miou, ious = self.metric.miou()
        if self.iter != 0 and self.iter % 10 == 0:
            logger.info(
                'Train, Epoch %d, Iter %d, LR %s, Loss %.4f, mIoU %.4f, IoUs %s' %
                (self.epoch, self.iter, self.lr, loss.item(),
                 miou, ious))

    def validate_batch(self, img, label):

        self.model.eval()
        with torch.no_grad():
            if self.gpu:
                img = img.cuda()
                label = label.cuda()

            pred = self.model(img)

            if not self.multilabel:
                prob = pred.softmax(dim=1)
                _, pred = torch.max(prob, dim=1)
            else:
                pred = pred.sigmoid()

            # if self.iter % 20 == 0:
            #     for i in range(4):
            #         cv2.imwrite(
            #             osp.join(self.workdir, 'vpred%d_%d.png' % (i, self.iter)),
            #             255 * pred_label.cpu().numpy()[i].astype(np.uint8))

            self.metric.add(pred.cpu().numpy(), label.cpu().numpy())
            miou, ious = self.metric.miou()
            logger.info('Validate, mIoU %.4f, IoUs %s' % (miou, ious))

    @staticmethod
    def test_preprocess(img, size):

        h, w = size
        resized_img = F.interpolate(img, size=size, mode='bilinear', align_corners=True)

        pad_h = int(np.ceil(h / 32) * 32) + 1 - h
        pad_w = int(np.ceil(w / 32) * 32) + 1 - w

        padding = (0, pad_w, 0, pad_h)

        padded_img = F.pad(resized_img, padding, 'constant', 0)

        return padded_img, (pad_h, pad_w)

    @staticmethod
    def test_postprocess(prob, pad_size, label_size):
        pad_h, pad_w = pad_size
        l_h, l_w = label_size

        prob = prob[:, :, :-pad_h, :-pad_w]

        # import pdb
        # pdb.set_trace()

        prob = F.interpolate(prob, size=label_size, mode='bilinear', align_corners=True)

        return prob

    def test_batch(self, img, label):
        self.model.eval()
        with torch.no_grad():

            if self.test_cfg:
                scales = self.test_cfg.get('scales', [1.0])
                flips = self.test_cfg.get('flips', [3])
                biases = self.test_cfg.get('bias', [0.0])
            else:
                scales = [1.0]
                flip = False
                biases = [0.0]

            assert len(scales) == len(biases)

            n, c, h, w = img.size()
            probs = []
            for i, (scale, bias) in enumerate(zip(scales, biases)):
                new_h, new_w = int(h * scale + bias), int(w * scale + bias)

                new_img, pad_size = self.test_preprocess(img, (new_h, new_w))

                # import pdb
                # pdb.set_trace()

                if self.gpu:
                    new_img = new_img.cuda()
                    label = label.cuda()

                prob = self.model(new_img).sigmoid()
                prob = self.test_postprocess(prob, pad_size, label.shape[-2:])

                # import pdb
                # pdb.set_trace()

                probs.append(prob)

                for f in flips:
                    flip_img = new_img.flip(f)
                    flip_prob = self.model(flip_img).sigmoid()
                    prob = flip_prob.flip(f)
                    prob = self.test_postprocess(prob, pad_size, label.shape[-2:])
                    probs.append(prob)

            # import pdb
            # pdb.set_trace()

            prob = torch.stack(probs, dim=0).mean(dim=0)

            self.metric.add(prob.cpu().numpy(), label.cpu().numpy())
            miou, ious = self.metric.miou()
            logger.info('Test, mIoU %.4f, IoUs %s' % (miou, ious))

    def judge_epoch(self, conf_thresholds=None, iou_thresholds=None):
        c_l, i_l, s_l, cl_l = len(conf_thresholds), \
                              len(iou_thresholds), \
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

            res = self.judge_batch(img,
                                   label,
                                   conf_thresholds=conf_thresholds,
                                   iou_thresholds=iou_thresholds,
                                   ori_img=ori_img,
                                   sample_id=sample_id)
            total_res[:, :, sample_id, :, :] = res

        total_res = total_res.sum(axis=2)

        for conf_idx, conf_thres in enumerate(conf_thresholds):
            for iou_idx, iou_thres in enumerate(iou_thresholds):
                # tp, tn, fp, fn = self.ana_tpfn(gt,
                #                                total_res[conf_idx, iou_idx, :])
                for c in range(self.ncls):
                    tp, fp, tn, fn = total_res[conf_idx, iou_idx, c]
                    p, r = self.ana_pr(tp, tn, fp, fn)
                    tpr, fpr = self.ana_roc(tp, tn, fp, fn)
                    precision[conf_idx, iou_idx, c] = p
                    recall[conf_idx, iou_idx, c] = r
                    tp_rates[conf_idx, iou_idx, c] = tpr
                    fp_rates[conf_idx, iou_idx, c] = fpr
                    tps[conf_idx, iou_idx, c] = tp
                    tns[conf_idx, iou_idx, c] = tn
                    fps[conf_idx, iou_idx, c] = fp
                    fns[conf_idx, iou_idx, c] = fn
        return tps, tns, fps, fns, precision, recall, tp_rates, fp_rates

    def judge_batch(self,
                    img, label,
                    conf_thresholds=None, iou_thresholds=None, ori_img=None, sample_id=None):

        res = np.zeros((len(conf_thresholds), len(iou_thresholds), self.ncls, 4))

        self.model.eval()
        with torch.no_grad():
            if self.gpu:
                img = img.cuda()
                label = label.cuda()

            pred = self.model(img)
            # prob = pred.softmax(dim=1)

        for idx_c, conf_th in enumerate(conf_thresholds):
            for idx_i, iou_th in enumerate(iou_thresholds):
                res[idx_c, idx_i] = self.judge_conf_map(img, pred, label,
                                                        conf_thres=conf_th,
                                                        iou_thres=iou_th,
                                                        ori_img=ori_img,
                                                        sample_id=sample_id)
        return res

    def judge_conf_map(self, img, prob, label, conf_thres=None, iou_thres=0.5, ori_img=None, sample_id=None):
        # tp fp tn fn
        tfpn = np.zeros((self.ncls, 4))

        prob = prob.sigmoid()
        prob[:, 1:, :, :] *= prob[:, 0, :, :]

        if conf_thres is None:
            conf_thres = 0.5

        # prob = prob[:, 1:, :, :]
        pred_label = torch.zeros_like(prob).long()
        pred_label[prob > conf_thres] = 1
        pred_label[label == 255] = 0

        ious = self.iou(pred_label.cpu().numpy(), label.cpu().numpy())

        for i in range(self.ncls):
            if (1 not in label[:, i]):
                if 1 not in pred_label[:, i]:
                    tfpn[i][2] += 1
                else:
                    tfpn[i][1] += 1
            else:
                if 1 not in pred_label[:, i]:
                    tfpn[i][3] += 1
                elif ious[i] > iou_thres:
                    tfpn[i][0] += 1
                else:
                    tfpn[i][3] += 1

            if self.show_fpfn or self.save_fpfn:
                fp, fn = tfpn[i][1], tfpn[i][3]
                self._show_fpfn(ori_img, i, pred_label[:, i], label[:, i], conf_thres, iou_thres, fp=fp, fn=fn, sample_id=sample_id)

        return tfpn

    def iou(self, pred, gt):
        pred = pred.transpose(1, 0, 2, 3)
        gt = gt.transpose(1, 0, 2, 3)
        # print(pred.shape, gt.shape)

        ious = []
        for i in range(self.ncls):
            sub_pred = pred[i]
            sub_gt = gt[i]

            mask = (sub_gt >= 0) & (sub_gt <= 1)
            sub_pred = sub_pred[mask]
            sub_gt = sub_gt[mask]

            interactions = np.sum(sub_pred & sub_gt)
            unions = np.sum(sub_pred | sub_gt)
            ious.append((interactions / (unions + 1e-9)))

        return ious

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

    def _show_fpfn(self, img, clsn, pred, label, conf_thres, iou_thres, fp=None, fn=None, sample_id=None):
        # label[label == 255] = 0

        # img = img[0][:, :, (2, 1, 0)].numpy()
        # img = self.draw_mask(img, pred.cpu().numpy()[0])
        # img = self.draw_mask(img, (label.cpu().numpy()[0] == 1), c=(255, 0, 0))

        if fp or fn:

            plt.figure(num=sample_id, figsize=(25, 8))
            plt.suptitle('cls %d & conf %.2f & iou %.2f & %s' % (clsn, conf_thres, iou_thres, 'fp' if fp else 'fn'))
            plt.subplot(1, 3, 1)
            plt.title('img')
            plt.imshow(img[0][:, :, (2, 1, 0)].int())
            plt.subplot(1, 3, 2)
            plt.title('pred')
            plt.imshow((255 * pred.cpu().numpy()[0]).astype(np.uint8))
            plt.subplot(1, 3, 3)
            plt.title('label')
            plt.imshow((255 * (label.cpu().numpy()[0] == 1)).astype(np.uint8))

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