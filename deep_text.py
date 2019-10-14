#!/usr/bin/env python
# -*- coding=utf-8 -*-

import cv2
import argparse
from model import CvEAST

def parse_cmdline_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', required=True, help='Path to input image')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_cmdline_flags()

    # Load the input image
    img = cv2.imread(args.input_image)

    # EAST model
    net = CvEAST('frozen_east_text_detection.pb')
