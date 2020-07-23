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
        description='divide datasets')
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

        print(re_train_idxs)

        train_simg_id, train_sann_id = process_restricted(
            train_data, ori_coco, re_train_idxs,
            train_simg_id, train_sann_id)
        train_simg_id = process_normal(
            train_data, folder, norm_files,
            norm_train_idxs, train_simg_id)

        val_simg_id, val_sann_id = process_restricted(
            val_data, ori_coco, re_val_idxs,
            val_simg_id, val_sann_id)
        val_simg_id = process_normal(
            val_data, folder, norm_files,
            norm_val_idxs, val_simg_id)

        process_csv(train_data, train_simg_id, train_sann_id)

    with open(args.output + '_train.json', 'w') as f:
        json.dump(train_data, f)

    with open(args.output + '_val.json', 'w') as f:
        json.dump(val_data, f)


def process_csv(data, simg_id, sann_id):
    data_root = 'data/SIXray'
    csv_file = 'data/SIXray/ImageSet/10/train.csv'
    img_prefix = 'data/SIXray/Image'
    divisor = 52515
    spec_class = [2, 5]

    lines = open(csv_file).readlines()
    ridxs = np.random.permutation(range(len(lines)))

    print(simg_id, sann_id)

    P_num = 0
    N_num = 0
    for i in tqdm(ridxs):
        if i == 0:
            continue

        line = lines[i]
        _data = line.split(',')
        name = _data[0]
        cls_data = [int(_) for _ in _data[1:]]
        num = int(name[1:])

        if name[0] == 'P':
            path = os.path.join(img_prefix, name + '.jpg')
        else:
            path = os.path.join(
                data_root,
                '%d' % ((num - 1) // divisor),
                name + '.jpg')

        img = cv2.imread(path)
        if img is None:
            continue

        h, w = img.shape[:2]
        if h > 513 or w > 513:
            continue

        if P_num == 200 and N_num == 800:
            break

        ann_incre = 0
        for j, l in enumerate(cls_data):
            if j + 1 in spec_class and l == 1:
                data['annotations'].append({
                    'id': sann_id,
                    'image_id': simg_id,
                    'category_id': 3 if j + 1 == 2 else j + 1,
                    'iscrowd': 0,
                    'segmentation': [],
                    'area': 100.0,
                    'bbox': [0, 0, 2, 2],
                    'minAreaRect': []
                })

                sann_id += 1
                ann_incre += 1

        if ann_incre > 0:
            P_num += 1
        elif N_num == 800:
            continue
        else:
            N_num += 1

        data['images'].append({
            'coco_url': '',
            'data_captured': '',
            'file_name': path,
            'flickr_url': '',
            'id': simg_id,
            'height': h,
            'width': w,
            'license': 1
        })

        simg_id += 1

        print(N_num, P_num)

    print(simg_id, sann_id)


def process_restricted(data, coco, idxs, simg_id, sann_id):

    for i in tqdm(idxs):
        i = int(i)
        img_info = coco.loadImgs(i)
        ann_ids = coco.getAnnIds(imgIds=[i])
        ann_infos = coco.loadAnns(ann_ids)

        img_info[0]['id'] = simg_id

        img_info[0]['file_name'] = osp.join('data/tianchi/jinnan2_round2_train_20190401/restricted', img_info[0]['file_name'])
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
            'file_name': osp.join('data/tianchi/jinnan2_round2_train_20190401/normal', path),
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
