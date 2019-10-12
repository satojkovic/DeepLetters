#!/usr/bin/env python
# -*- coding=utf-8 -*-

import cv2

if __name__ == "__main__":
    # East model
    print('** Load EAST model **')
    net = cv2.dnn.readNet('frozen_east_text_detection.pb')
