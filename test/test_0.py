# -*- coding: utf-8 -*-

# pylint: disable=missing-docstring
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function
import numpy as np




import tensorflow  as tf
#import tensorboard as tb

#import numpy
import os
#from tensorboard.summary import image
#import math
#import cmath
#import random


from PIL import Image
import numpy as np
import tensorflow as tf
import glob

def read_and_decode(filename_queue):
    """Read from tfrecords file and decode and normalize the image data."""
    reader = tf.TFRecordReader()
    _, serialized_exmaple = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_exmaple,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        },
    )

    image = tf.decode_raw(features['image'], tf.uint8)
    image.set_shape([100 * 100])

    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    label = tf.cast(features['label'], tf.int32)

    return image, label



folder_path = r'C:\Users\I347798\git\project_tomato\test\OUTPUT_DIR\train-00000-of-00001.tfrecords'
filename_queue = tf.train.string_input_producer(
    [folder_path], num_epochs=1)
print(read_and_decode(filename_queue))

#get_all_records(r'C:\Users\I347798\git\project_tomato\test\OUTPUT_DIR\train-00000-of-00001')

#read_and_decode(r'C:\Users\I347798\git\project_tomato\test\OUTPUT_DIR\train-00000-of-00001')