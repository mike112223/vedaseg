import argparse
import os
import pdb
import sys

sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../vedaseg'))

import cv2
import torch
import numpy as np

from vedaseg.datasets import build_dataset
from vedaseg.datasets.transforms.builder import build_transform
from vedaseg.dataloaders import build_dataloader
from vedaseg.models import build_model
import vedaseg.utils as utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('save_path')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg_fp = args.config
    checkpoint = args.checkpoint
    # save_path = args.save_path

    _, fullname = os.path.split(cfg_fp)

    cfg = utils.Config.fromfile(cfg_fp)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu_id']

    # os.makedirs(save_path, exist_ok=True)

    tf = build_transform(cfg['data']['val']['transforms'])
    datasets = build_dataset(cfg['data']['val']['dataset'], dict(transform=tf, infer=True))
    loader = build_dataloader(cfg['data']['val']['loader'], dict(dataset=datasets))

    model = build_model(cfg['model'])
    model.cuda()
    model.load_state_dict(torch.load(checkpoint)['state_dict'])
    model.eval()
    print('infer')
    with torch.no_grad():
        count = 0
        for images, masks, init_imgs in loader:
            print(count)
            count += 1
            prob = model(images.cuda()).softmax(dim=1)
            for i in range(prob.shape[0]):
                c_prob = prob[i]
                c_img = images[i]
                c_mask = masks[i]
                i_img = init_imgs[i]
                c_img = c_img.cpu().numpy()
                c_img = (c_img - np.min(c_img)) / (np.max(c_img) - np.min(c_img))
                # pdb.set_trace()
                cv2.imshow('mask', 255 * c_mask.cpu().numpy().astype(np.uint8))
                cv2.imshow('img', c_img.transpose(1, 2, 0))
                cv2.imshow('init_img', i_img.cpu().numpy())
                cv2.imshow('pred', (255 * c_prob[1].cpu().numpy()).astype(np.uint8))
                cv2.waitKey()

            # pk.dump(images, open(f'{save_path}/{count}_1.pkl', 'wb'))
            # pk.dump(prob, open(f'{save_path}/{count}_2.pkl', 'wb'))


if __name__ == '__main__':
    main()
