#!/usr/bin/env python
# -*- coding=utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.slim as slim

INPUT_SIZE = 700
INPUT_CH = 3


def forward(input_images):
    end_points = {}
    
    # VGG-16 (conv1_1 through conv4_3)
    with slim.arg_scope([slim.conv2d], kernel_size=(3, 3)):
        # 700 x 700 x 3
        end_point = 'conv1_1'
        net = slim.conv2d(input_images, 64)
        end_points[end_point] = net
        print(end_point, ':', net.get_shape())
        # 700 x 700 x 64
        end_point = 'conv1_2'
        net = slim.conv2d(net, 64)
        end_points[end_point] = net
        print(end_point, ':', net.get_shape())
        # 700 x 700 x 64
        end_point = 'pool1'
        net = slim.max_pool2d(net, (2, 2), padding='SAME')
        end_points[end_point] = net
        print(end_point, ':', net.get_shape())
        # 350 x 350 x 64
        end_point = 'conv2_1'
        net = slim.conv2d(net, 128)
        end_points[end_point] = net
        print(end_point, ':', net.get_shape())
        # 350 x 350 x 128
        end_point = 'conv2_2'
        net = slim.conv2d(net, 128)
        end_points[end_point] = net
        print(end_point, ':', net.get_shape())
        # 350 x 350 x 128
        end_point = 'pool2'
        net = slim.max_pool2d(net, (2, 2), padding='SAME')
        end_points[end_point] = net
        print(end_point, ':', net.get_shape())
        # 175 x 175 x 128
        end_point = 'conv3_1'
        net = slim.conv2d(net, 256)
        end_points[end_point] = net
        print(end_point, ':', net.get_shape())
        # 175 x 175 x 256
        end_point = 'conv3_2'
        net = slim.conv2d(net, 256)
        end_points[end_point] = net
        print(end_point, ':', net.get_shape())
        # 175 x 175 x 256
        end_point = 'conv3_3'
        net = slim.conv2d(net, 256)
        end_points[end_point] = net
        print(end_point, ':', net.get_shape())
        # 175 x 175 x 256
        end_point = 'pool3'
        net = slim.max_pool2d(net, (2, 2), padding='SAME')
        end_points[end_point] = net
        print(end_point, ':', net.get_shape())
        # 88 x 88 x 256
        end_point = 'conv4_1'
        net = slim.conv2d(net, 512)
        end_points[end_point] = net
        print(end_point, ':', net.get_shape())
        # 88 x 88 x 512
        end_point = 'conv4_2'
        net = slim.conv2d(net, 512)
        end_points[end_point] = net
        print(end_point, ':', net.get_shape())
        # 88 x 88 x 512
        end_point = 'conv4_3'
        net = slim.conv2d(net, 512)
        end_points[end_point] = net
        print(end_point, ':', net.get_shape())
        # 88 x 88 x 512
        end_point = 'pool4'
        net = slim.max_pool2d(net, (2, 2), padding='SAME')
        end_points[end_point] = net
        print(end_point, ':', net.get_shape())
        # 44 x 44 x 512
        end_point = 'conv5_1'
        net = slim.conv2d(net, 512)
        end_points[end_point] = net
        print(end_point, ':', net.get_shape())
        # 44 x 44 x 512
        end_point = 'conv5_2'
        net = slim.conv2d(net, 512)
        end_points[end_point] = net
        print(end_point, ':', net.get_shape())
        # 44 x 44 x 512
        end_point = 'conv5_3'
        net = slim.conv2d(net, 512)
        end_points[end_point] = net
        print(end_point, ':', net.get_shape())
        # 44 x 44 x 512
        end_point = 'pool5'
        net = slim.max_pool2d(net, (3, 3), stride=1, padding='SAME')
        end_points[end_point] = net
        print(end_point, ':', net.get_shape())
        # 44 x 44 x 512

    logits = net
    return logits, end_points