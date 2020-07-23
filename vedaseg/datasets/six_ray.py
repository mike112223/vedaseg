
import os
import os.path as osp

import cv2
import torch
import numpy as np
from pycocotools.coco import COCO
from .base import BaseDataset
from .registry import DATASETS


@DATASETS.register_module
class SIXRayDataset(BaseDataset):

    def __init__(self,
                 csv_file,
                 data_root=None,
                 img_prefix='',
                 spec_class=None,
                 transform=None,
                 infer=False,
                 extra_super=False,
                 rand_same=False,
                 label_epsilon=-1):
        self.csv_file = csv_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.spec_class = spec_class
        self.transform = transform
        self.infer = infer
        self.extra_super = extra_super
        self.rand_same = rand_same
        self.label_epsilon = label_epsilon

        # self.bins = [
        #     52515, 105030, 157545, 210060, 262575,
        #     315090, 367605, 420120, 472635, 525150,
        #     577665, 630180, 682695, 735210, 787725,
        #     840240, 892755, 945270, 997785, 1050302
        # ]
        self.divisor = 52515

        if isinstance(self.spec_class, int):
            self.spec_class = [self.spec_class]

        if self.data_root is not None:
            if not osp.isabs(self.csv_file):
                self.csv_file = osp.join(self.data_root, self.csv_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)

        # load annotations (and proposals)
        self.data_infos = self.load_csvs(self.csv_file)

        self._set_group_flag()

    def load_csvs(self, csv_file):
        lines = open(csv_file).readlines()
        data_infos = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            info = {}

            data = line.split(',')
            name = data[0]
            cls_data = [int(_) for _ in data[1:]]
            num = int(name[1:])

            if name[0] == 'P':
                info['filename'] = os.path.join(self.img_prefix, name + '.jpg')
            else:
                info['filename'] = os.path.join(
                    self.data_root,
                    '%d' % ((num - 1) // self.divisor),
                    name + '.jpg')

            label = []
            for j, l in enumerate(cls_data):
                if j + 1 in self.spec_class and l == 1:
                    label.append(1)
                else:
                    label.append(0)

            info['label'] = label

            data_infos.append(info)

        return data_infos

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.norm_flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            data_info = self.data_infos[i]
            if data_info['filename'].split('/')[-1][0] == 'N':
                self.norm_flag[i] = 1
            else:
                self.norm_flag[i] = 0

    def __len__(self):
        return len(self.data_infos)

    def _get_img_info(self, idx):
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        # print(idx, img_info)

        img = cv2.imread(img_info['filename']).astype(np.float32)
        ori_img = img.copy()

        dmasks = self.draw_mask(img, ann_info)

        return ori_img, img, dmasks, img_info['file_name']

    def _rand_another(self):
        return np.random.choice(range(len(self)))

    def __getitem__(self, idx):

        img = None
        while img is None:
            data = self.data_infos[idx]
            label = data['label']

            img = cv2.imread(data['filename'])

            if img is None:
                print('empty img!!! {}'.format(data['filename']))
                idx = self._rand_another()

        img = img.astype(np.float32)

        ori_img = img.copy()

        img, _ = self.process(img, None)

        if self.infer:
            return img, label, ori_img, data['filename']
        else:
            return img, label
