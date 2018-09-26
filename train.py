#!/usr/bin/env python
# -*- coding=utf-8 -*-

import tensorflow as tf
import deep_text


def main():
    with tf.Graph().as_default():
        input_images = tf.placeholder(
            tf.float32, 
            shape=(None, deep_text.INPUT_SIZE, deep_text.INPUT_SIZE, deep_text.INPUT_CH),
            name='input_images'
        )

        # DeepText model
        print('** DeepText model **')
        end_points = deep_text.forward(input_images)


if __name__ == "__main__":
    main()
