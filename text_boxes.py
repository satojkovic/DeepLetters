#!/usr/bin/env python
# -*- coding=utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.slim as slim

INPUT_SIZE = 300
INPUT_CH = 3


def forward(input_images):
    end_points = {}
    
    # VGG-16 (conv1_1 through conv4_3)
    with slim.arg_scope([slim.conv2d], kernel_size=(3, 3)):
        end_point = 'conv1_1'
        net = slim.conv2d(input_images, 64)
        end_points[end_point] = net
        print(end_point, ':', net.get_shape())

    logits = net
    return logits, end_points