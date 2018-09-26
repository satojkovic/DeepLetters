#!/usr/bin/env python
# -*- coding=utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

INPUT_SIZE = 300
INPUT_CH = 3

ANCHOR_SCALES = [8, 16, 32]
ANCHOR_RATIOS = [0.5, 1, 2]
FEAT_STRIDE = [16,]


def _whctrs(anchor):
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1), 
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors


def _ratio_enum(anchor, ratios):
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    # The ratio of width are [0.5, 1, 2]
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * np.asarray(scales)
    hs = h * np.asarray(scales)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def generate_anchors(anchor_scales, anchor_ratios):
    base_anchor = np.array([1, 1, anchor_scales[1], anchor_scales[1]]) - 1
    ratio_anchors = _ratio_enum(base_anchor, anchor_ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], anchor_scales) 
                         for i in range(ratio_anchors.shape[0])])
    print('anchors = ', anchors)
    return anchors


def RPN(base_feat):
    rpn_conv = slim.conv2d(base_feat, 512, kernel_size=(3, 3), stride=1, padding='SAME')

    # rpn classification score
    num_scores = len(ANCHOR_SCALES) * len(ANCHOR_RATIOS) * 2
    rpn_cls_score = slim.conv2d(rpn_conv, num_scores, kernel_size=(1, 1), stride=1, padding='SAME')
    rpn_cls_score = slim.softmax(rpn_cls_score)
    print('rpn_cls_score:', rpn_cls_score.get_shape())

    # rpn offsets to the anchor boxes
    num_anchor_boxes = len(ANCHOR_SCALES) * len(ANCHOR_RATIOS) * 4
    rpn_bbox_pred = slim.conv2d(base_feat, num_anchor_boxes, kernel_size=(1, 1), stride=1, padding='SAME')
    print('rpn_bbox_pred:', rpn_bbox_pred.get_shape())

    # Generate anchors 
    anchors = generate_anchors(ANCHOR_SCALES, ANCHOR_RATIOS)

    return

def frcnn_base(input_images):
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
    base_feat = end_points['conv5_3']
    return base_feat, end_points


def forward(input_images):
    # base features
    base_feat, end_points = frcnn_base(input_images)

    #rois, rpn_loss_cls, rpn_loss_bbox = RPN(base_feat)
    RPN(base_feat)

    return end_points
