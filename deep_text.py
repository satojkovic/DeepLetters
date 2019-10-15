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

        ratio_w, ratio_h, indices, boxes = net.predict(frame)
        for i in indices:
            vertices = cv2.boxPoints(boxes[i[0]])
            for j in range(4):
                vertices[j][0] *= ratio_w
                vertices[j][1] *= ratio_h
            for j in range(4):
                p1 = (vertices[j][0], vertices[j][1])
                p2 = (vertices[(j + 1) % 4][0], vertices[(j + 1) % 4][1])
                cv2.line(frame, p1, p2, (0, 255, 0), 1)

        cv2.imshow('Text Detection', frame)