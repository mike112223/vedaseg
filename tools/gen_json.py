import argparse
import sys
import os
import os.path as osp
import json
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../vedaseg'))

import cv2
from tqdm import tqdm
import numpy as np
from pycocotools.coco import COCO

from vedaseg import utils


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a semantic segmentatation model')
    parser.add_argument('--folder', nargs='+')
    parser.add_argument('--json', nargs='+')
    parser.add_argument('--seed', default=1234)
    parser.add_argument('--ratio', default=0.8)
    parser.add_argument('--output', default='mix')
    args = parser.parse_args()
    return args


def gen_json(args):

    train_data = {}
    val_data = {}
    train_simg_id, val_simg_id = 0, 0
    train_sann_id, val_sann_id = 0, 0

    for folder, jsonp in zip(args.folder, args.json):
        norm_files = os.listdir(osp.join(folder, 'normal'))
        re_files = os.listdir(osp.join(folder, 'restricted'))

        ori_data = json.load(open(osp.join(folder, jsonp)))
        ori_coco = COCO(osp.join(folder, jsonp))

        if len(train_data.keys()) == 0:
            for k, v in ori_data.items():
                if k not in ('images', 'annotations'):
                    train_data[k] = v
                    val_data[k] = v
                else:
                    train_data[k] = []
                    val_data[k] = []

        img_ids = ori_coco.getImgIds()
        re_train_idxs, re_val_idxs = split(img_ids, args.ratio)
        norm_train_idxs, norm_val_idxs = split(norm_files, args.ratio)

        train_simg_id, train_sann_id = process_restricted(
            train_data, ori_coco, re_train_idxs,
            train_simg_id, train_sann_id)
        train_simg_id = process_normal(
            train_data, folder, norm_files,
            norm_train_idxs, train_simg_id)

        val_simg_id, val_sann_id = process_restricted(
            val_data, ori_coco, re_val_idxs,
            val_simg_id, val_sann_id)
        train_simg_id = process_normal(
            val_data, folder, norm_files,
            norm_val_idxs, val_simg_id)

    with open(args.output + '_train.json', 'w') as f:
        json.dump(train_data, f)

    with open(args.output + '_val.json', 'w') as f:
        json.dump(val_data, f)


def process_restricted(data, coco, idxs, simg_id, sann_id):

    for i in tqdm(idxs):
        i = int(i)
        img_info = coco.loadImgs(i)
        ann_ids = coco.getAnnIds(imgIds=[i])
        ann_infos = coco.loadAnns(ann_ids)

        img_info[0]['id'] = simg_id

        img_info[0]['file_name'] = osp.join('restricted', img_info[0]['file_name'])
        data['images'].extend(img_info)

        for ann in ann_infos:
            ann['id'] = sann_id
            sann_id += 1

            ann['image_id'] = simg_id

        data['annotations'].extend(ann_infos)
        simg_id += 1

    return simg_id, sann_id


def process_normal(data, folder, files, idxs, simg_id):

    for i in tqdm(idxs):
        path = files[i]
        img = cv2.imread(osp.join(folder, 'normal', path))
        h, w = img.shape[:2]

        data['images'].append({
            'coco_url': '',
            'data_captured': '',
            'file_name': osp.join('normal', path),
            'flickr_url': '',
            'id': simg_id,
            'height': h,
            'width': w,
            'license': 1
        })

        simg_id += 1

    return simg_id


def split(files, ratio):
    l = len(files)
    ridxs = np.random.permutation(l)
    train_idxs = ridxs[:int(l * ratio)]
    val_idxs = ridxs[int(l * ratio):]
    return train_idxs, val_idxs


def main():
    args = parse_args()

    utils.set_random_seed(args.seed)

    gen_json(args)


if __name__ == '__main__':
    main()
