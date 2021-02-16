#!/usr/bin/env python
# -*- coding=utf-8 -*-

import argparse
import scipy.io as sio
import numpy as np
from tqdm import tqdm
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
from PIL import Image
from object_detection.utils import dataset_util

class SynthText:
    def __init__(self, gt_mat_path):
        self.gt_mat_path = gt_mat_path
        self.gt_mat_dir = os.path.dirname(os.path.abspath(gt_mat_path))
        self.gt_mat = self._load_gt_mat()
        self._preproc()

    def train_test_split(self, train_ratio=0.75):
        num_indices = len(self.wordBB)
        train_indices = np.random.choice(num_indices, int(num_indices * train_ratio), replace=False)
        test_indices = list(set(np.arange(num_indices)) ^ set(train_indices))
        return train_indices, test_indices

    def _load_gt_mat(self):
        return sio.loadmat(self.gt_mat_path)

    def _preproc(self):
        self._txt = self._preproc_gt_txt()
        self._indices = self._get_indices()
        self._wordBB = self.gt_mat['wordBB'][0][self._indices]
        self._txt = self._txt[self._indices]
        self._imnames = self.gt_mat['imnames'][0][self._indices]
        self._remove_invalid_boxes()

    def _remove_invalid_boxes(self):
        self.wordBB = []
        self.txt = []
        self.imnames = []
        for index in range(len(self._wordBB)):
            wordBB = []
            txt = []
            for bindex in range(self._wordBB[index].shape[-1]):
                xmin = self._wordBB[index][0][0][bindex]
                ymin = self._wordBB[index][1][0][bindex]
                xmax = self._wordBB[index][0][2][bindex]
                ymax = self._wordBB[index][1][2][bindex]
                if ymin > ymax or xmin > xmax:
                    continue
                wordBB.append([xmin, ymin, xmax, ymax])
                txt.append(self._txt[index][bindex])
            if len(wordBB) != 0:
                self.wordBB.append(wordBB)
                self.txt.append(txt)
                self.imnames.append(self._imnames[index])

    def _preproc_gt_txt(self):
        processed_txt = []
        for i in range(self.gt_mat['txt'].shape[-1]):
            tmp = [w.strip() for words in self.gt_mat['txt'][0][i][:] for w in words.split('\n')]
            processed_txt.append(tmp)
        return np.asarray(processed_txt)

    def _get_indices(self):
        indices = []
        for index in range(self.gt_mat['imnames'].shape[-1]):
            if self.gt_mat['wordBB'][0][index].shape[-1] == len(self._txt[index]):
                indices.append(index)
        return indices

def create_tfrecord(gt_mat_dir, filename, wordBB, txt):
    img = Image.open(os.path.join(gt_mat_dir, filename))
    width, height = img.size
    with tf.gfile.GFile(os.path.join(gt_mat_dir, filename), 'rb') as fid:
        encoded_jpg = fid.read()
    filename = os.path.join(gt_mat_dir, filename)
    filename = filename.encode('utf8')
    image_format = b'jpg'

    xmins, ymins = [], []
    xmaxs, ymaxs = [], []
    classes_text = []
    classes = []

    for index in range(len(wordBB)):
        xmin = wordBB[index][0] / width
        ymin = wordBB[index][1] / height
        xmax = wordBB[index][2] / width
        ymax = wordBB[index][3] / height
        if xmin < 1 and ymin < 1 and xmax < 1 and ymax < 1:
            xmins.append(xmin)
            ymins.append(ymin)
            xmaxs.append(xmax)
            ymaxs.append(ymax)
            classes_text.append('Text'.encode('utf8'))
            classes.append(1)

    if len(xmins) != 0:
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
    else:
        return None
    return tf_example

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_mat_path', required=True, help='Path to gt.mat')
    args = parser.parse_args()

    synth_text = SynthText(args.gt_mat_path)
    print('charBB: {}, wordBB: {}'.format(len(synth_text.gt_mat['charBB'][0]), len(synth_text.gt_mat['wordBB'][0])))
    print('Text sample: {}'.format(synth_text.txt[0]))
    print('Preprocessed: len(wordBB)={}, len(txt)={}, len(imnames)={}'.format(
        len(synth_text.wordBB), len(synth_text.txt), len(synth_text.imnames)))

    train_indices, test_indices = synth_text.train_test_split()
    print('Train: {}, Test: {}'.format(len(train_indices), len(test_indices)))

    train_writer = tf.python_io.TFRecordWriter('synth_text_train.tfrecord')
    for index in tqdm(train_indices, total=len(train_indices)):
        filename = synth_text.imnames[index][0]
        wordBB = synth_text.wordBB[index]
        txt = synth_text.txt[index]
        tf_record = create_tfrecord(synth_text.gt_mat_dir, filename, wordBB, txt)
        if tf_record is not None:
            train_writer.write(tf_record.SerializeToString())
    train_writer.close()

    seen = set()
    test_writer = tf.python_io.TFRecordWriter('synth_text_test.tfrecord')
    for index in tqdm(test_indices, total=len(test_indices)):
        filename = synth_text.imnames[index][0]
        if filename in seen:
            continue
        seen.add(filename)
        wordBB = synth_text.wordBB[index]
        txt = synth_text.txt[index]
        tf_record = create_tfrecord(synth_text.gt_mat_dir, filename, wordBB, txt)
        if tf_record is not None:
            test_writer.write(tf_record.SerializeToString())
    print('Wrote synth_text_test.tfrecord:', len(seen))
    test_writer.close()