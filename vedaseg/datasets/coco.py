
import os
import os.path as osp

import cv2
import torch
import numpy as np
from pycocotools.coco import COCO
from .base import BaseDataset
from .registry import DATASETS


@DATASETS.register_module
class CocoDataset(BaseDataset):

    def __init__(self,
                 ann_file,
                 data_root=None,
                 img_prefix='',
                 spec_class=None,
                 filter_empty_gt=True,
                 transform=None,
                 infer=False,
                 extra_super=False):
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.spec_class = spec_class
        self.filter_empty_gt = filter_empty_gt
        self.transform = transform
        self.infer = infer
        self.extra_super = extra_super

        if isinstance(self.spec_class, int):
            self.spec_class = [self.spec_class]

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)

        # load annotations (and proposals)
        self.data_infos = self.load_annotations(self.ann_file)

    def __len__(self):
        return len(self.data_infos)

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.getImgIds()

        data_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = os.path.join(self.img_prefix, info['file_name'])
            data_infos.append(info)
        return data_infos

    def get_ann_info(self, idx):
        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.data_infos):
            if self.filter_empty_gt and self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info):

            if self.spec_class is not None and ann['category_id'] not in self.spec_class:
                continue

            if ann['category_id'] not in self.cat_ids:
                continue

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        # import pdb
        # pdb.set_trace()

        return ann

    def draw_mask(self, img, ann_info):
        if self.spec_class is not None:
            dmasks = np.zeros((1, img.shape[0], img.shape[1]), np.uint8)
            for mask in ann_info['masks']:
                mask = np.asarray(mask).reshape(-1, 1, 2).astype(np.int32)
                cv2.drawContours(dmasks[0], [mask], -1, 1, cv2.FILLED)
        else:
            c = len(self.cat_ids)
            dmasks = np.zeros((c, img.shape[0], img.shape[1]), np.uint8)
            for mask, label in zip(ann_info['masks'], ann_info['labels']):
                mask = np.asarray(mask).reshape(-1, 1, 2).astype(np.int32)
                cv2.drawContours(dmasks[label], [mask], -1, 1, cv2.FILLED)

            if self.extra_super:
                fmask = np.max(dmasks, axis=0)[None, :, :]
                dmasks = np.concatenate([fmask, dmasks])

        return dmasks

    def __getitem__(self, idx):
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        # print(idx, img_info)

        img = cv2.imread(img_info['filename']).astype(np.float32)
        ori_img = img.copy()

        # cv2.imwrite('workdir/debug/img_%d.png' % idx, ori_img.astype(np.uint8))

        dmasks = self.draw_mask(img, ann_info)

        img, mask = self.process(img, dmasks)

        if self.spec_class is not None:
            mask = mask.long()[0]
        else:
            mask = mask.long()

        if self.infer:
            return img, mask, ori_img
        else:
            return img, mask
