#!/usr/bin/env python
# -*- coding=utf-8 -*-

import cv2
import argparse
from model import CvEAST

def parse_cmdline_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to input image or video file')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_cmdline_flags()

    # Load EAST model
    net = CvEAST('frozen_east_text_detection.pb')

    # Open a video file or an image file
    cap = cv2.VideoCapture(args.input if args.input else 0)

    while cv2.waitKey(1) < 0:
        has_frame, frame = cap.read()
        if not has_frame:
            cv2.waitKey(0)
            break

        ratio_w, ratio_h, results = net.predict(frame)
        for ((start_x, start_y, end_x, end_y), text) in results:
            text = ''.join([c if ord(c) < 128 else '' for c in text]).strip()
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
            cv2.putText(frame, text, (start_x, start_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow('Text Detection', frame)