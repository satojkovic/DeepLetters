#!/usr/bin/env python
# -*- coding=utf-8 -*-

import argparse
import scipy.io as sio

def load_gt_mat(gt_mat_path):
    return sio.loadmat(gt_mat_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_mat_path', required=True, help='Path to gt.mat')
    args = parser.parse_args()

    gt_mat = load_gt_mat(args.gt_mat_path)
    print('charBB: {}, wordBB: {}'.format(len(gt_mat['charBB'][0]), len(gt_mat['wordBB'][0])))