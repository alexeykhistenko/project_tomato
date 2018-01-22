# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import os
import sys


TRAIN_FILE = 'train-00000-of-00001.tfrecords'
VALIDATION_FILE = 'validation-00000-of-00001.tfrecords'

IMAGE_SIZE = 100
NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1100
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 300


def read_from_file(file_path):
    # DON'T TOUCH THIS SHIT
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_path)
    features = tf.parse_single_example(
        serialized_example,

        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    
    
    