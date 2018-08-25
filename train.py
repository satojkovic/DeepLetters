#!/usr/bin/env python
# -*- coding=utf-8 -*-

import tensorflow as tf
import text_boxes


def main():
    with tf.Graph().as_default():
        input_images = tf.placeholder(
            tf.float32, 
            shape=(None, text_boxes.INPUT_SIZE, text_boxes.INPUT_SIZE, text_boxes.INPUT_CH),
            name='input_images'
        )

        # TextBoxes model
        print('** TextBoxes model **')
        logits, end_points = text_boxes.forward(input_images)


if __name__ == "__main__":
    main()
