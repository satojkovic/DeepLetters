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
        # 300 x 300 x 3
        end_point = 'conv1_1'
        net = slim.conv2d(input_images, 64)
        end_points[end_point] = net
        print(end_point, ':', net.get_shape())
        # 300 x 300 x 64
        end_point = 'conv1_2'
        net = slim.conv2d(net, 64)
        end_points[end_point] = net
        print(end_point, ':', net.get_shape())
        # 300 x 300 x 64
        end_point = 'pool1'
        net = slim.max_pool2d(net, (2, 2), padding='SAME')
        end_points[end_point] = net
        print(end_point, ':', net.get_shape())
        # 150 x 150 x 64
        end_point = 'conv2_1'
        net = slim.conv2d(net, 128)
        end_points[end_point] = net
        print(end_point, ':', net.get_shape())
        # 150 x 150 x 128
        end_point = 'conv2_2'
        net = slim.conv2d(net, 128)
        end_points[end_point] = net
        print(end_point, ':', net.get_shape())
        # 150 x 150 x 128
        end_point = 'pool2'
        net = slim.max_pool2d(net, (2, 2), padding='SAME')
        end_points[end_point] = net
        print(end_point, ':', net.get_shape())
        # 75 x 75 x 128
        end_point = 'conv3_1'
        net = slim.conv2d(net, 256)
        end_points[end_point] = net
        print(end_point, ':', net.get_shape())
        # 75 x 75 x 256
        end_point = 'conv3_2'
        net = slim.conv2d(net, 256)
        end_points[end_point] = net
        print(end_point, ':', net.get_shape())
        # 75 x 75 x 256
        end_point = 'conv3_3'
        net = slim.conv2d(net, 256)
        end_points[end_point] = net
        print(end_point, ':', net.get_shape())
        # 75 x 75 x 256
        end_point = 'pool3'
        net = slim.max_pool2d(net, (2, 2), padding='SAME')
        end_points[end_point] = net
        print(end_point, ':', net.get_shape())
        # 38 x 38 x 256
        end_point = 'conv4_1'
        net = slim.conv2d(net, 512)
        end_points[end_point] = net
        print(end_point, ':', net.get_shape())
        # 38 x 38 x 512
        end_point = 'conv4_2'
        net = slim.conv2d(net, 512)
        end_points[end_point] = net
        print(end_point, ':', net.get_shape())
        # 38 x 38 x 512
        end_point = 'conv4_3'
        net = slim.conv2d(net, 512)
        end_points[end_point] = net
        print(end_point, ':', net.get_shape())
        # 38 x 38 x 512
        end_point = 'pool4'
        net = slim.max_pool2d(net, (2, 2), padding='SAME')
        end_points[end_point] = net
        print(end_point, ':', net.get_shape())
        # 19 x 19 x 512
        end_point = 'conv5_1'
        net = slim.conv2d(net, 512)
        end_points[end_point] = net
        print(end_point, ':', net.get_shape())
        # 19 x 19 x 512
        end_point = 'conv5_2'
        net = slim.conv2d(net, 512)
        end_points[end_point] = net
        print(end_point, ':', net.get_shape())
        # 19 x 19 x 512
        end_point = 'conv5_3'
        net = slim.conv2d(net, 512)
        end_points[end_point] = net
        print(end_point, ':', net.get_shape())
        # 19 x 19 x 512
        end_point = 'pool5'
        net = slim.max_pool2d(net, (3, 3), stride=1, padding='SAME')
        end_points[end_point] = net
        print(end_point, ':', net.get_shape())
        # 19 x 19 x 512
        end_point = 'conv6'
        net = slim.conv2d(net, 1024, rate=6)
        end_points[end_point] = net
        print(end_point, ':', net.get_shape())
        # 19 x 19 x 1024
        end_point = 'conv7'
        net = slim.conv2d(net, 1024, kernel_size=[1, 1])
        end_points[end_point] = net
        print(end_point, ':', net.get_shape())
        # 19 x 19 x 1024
        end_point = 'conv8'
        net = slim.conv2d(net, 256, kernel_size=[1, 1])
        net = tf.pad(net, [[0, 0], [1,1], [1,1], [0, 0]])
        net = slim.conv2d(net, 512, stride=2, padding='VALID')
        end_points[end_point] = net
        print(net.get_shape())
        # 10 x 10 x 512
        end_point = 'conv9'
        net = slim.conv2d(net, 128, kernel_size=[1, 1])
        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
        net = slim.conv2d(net, 256, stride=2, padding='VALID')
        end_points[end_point] = net
        print(net.get_shape())
        # 5 x 5 x 256
        end_point = 'conv10'
        net = slim.conv2d(net, 128, kernel_size=[1, 1])
        net = slim.conv2d(net, 256, padding='VALID')
        end_points[end_point] = net
        print(net.get_shape())
        # 3 x 3 x 256
        end_point = 'global'
        net = slim.conv2d(net, 128, kernel_size=[1, 1])
        net = slim.conv2d(net, 256, padding='VALID')
        end_points[end_point] = net
        print(net.get_shape())
        # 1 x 1 x 256

    logits = net
    return logits, end_points