#!/usr/bin/env python
# -*- coding=utf-8 -*-

import cv2
import argparse
from model import CvEAST
import tensorflow as tf
import os
from object_detection.utils import ops as utils_ops
import numpy as np
import pytesseract

def parse_cmdline_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to input image or video file')
    parser.add_argument('--inference_graph_dir_path', required=True, help='Path to inference graph directory')
    parser.add_argument('--detection_th', type=float, required=True, help='Threshold value for detection boxes')
    return parser.parse_args()

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[1], image.shape[2])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: image})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.int64)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

if __name__ == "__main__":
    args = parse_cmdline_flags()

    # Load SSD model
    PATH_TO_FROZEN_GRAPH = os.path.join(args.inference_graph_dir_path, 'frozen_inference_graph.pb')
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as f:
            od_graph_def.ParseFromString(f.read())
            tf.import_graph_def(od_graph_def, name='')

    # Open a video file or an image file
    cap = cv2.VideoCapture(args.input if args.input else 0)

    while cv2.waitKey(1) < 0:
        has_frame, frame = cap.read()
        if not has_frame:
            cv2.waitKey(0)
            break

        im_height, im_width, _ = frame.shape
        frame = np.expand_dims(frame, axis=0)
        output_dict = run_inference_for_single_image(frame, detection_graph)

        for i in range(output_dict['num_detections']):
            if output_dict['detection_scores'][i] < args.detection_th:
                continue
            ymin, xmin, ymax, xmax = output_dict['detection_boxes'][i]
            left, right, top, bottom = int(xmin * im_width), int(xmax * im_width), int(ymin * im_height), int(ymax * im_height)
            cv2.rectangle(frame[0], (left, top), (right, bottom), (0, 0, 255), 2)

            # Text recognition
            text = pytesseract.image_to_string(frame[0][top:bottom, left:right], config=r'--psm 7')
            text = ''.join([c if ord(c) < 128 else '' for c in text]).strip()
            cv2.putText(frame[0], text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

        cv2.imshow('Text Detection', frame[0])