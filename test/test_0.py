# -*- coding: utf-8 -*-

# pylint: disable=missing-docstring
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function
import numpy as np



import keras
from keras.preprocessing.image import ImageDataGenerator
import tensorflow  as tf
#import tensorboard as tb

#import numpy
import os
from tensorboard.summary import image
from test.test import batch_size



#import math
#import cmath
#import random


"""Глобальные переменные"""

IMAGE_SIZE = 100

NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1100
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 300
batch_size = 12
#CURRENT_DIRECTORY = os.getcwd()

CURRENT_DIRECTORY = "C:/Users/I347798/git/project_tomato/"
IMAGES_DIRECTORY = "images/"

with open(CURRENT_DIRECTORY + "lables.txt", "r") as lables:
    for i in lables.readlines():
        path = CURRENT_DIRECTORY + IMAGES_DIRECTORY + i.rstrip("\n")
    #    print(path)
        aaa = os.listdir(path)
    #    print(aaa)
        for x in aaa:
            image = tf.image.decode_jpeg(x, channels=3)
            tensor_image = [image]
#            with open(CURRENT_DIRECTORY + "test.txt", "w") as f:
#                f.write(image)
            print(image)
           
#datagen = ImageDataGenerator(rescale=1. / 255)
'''
path = CURRENT_DIRECTORY + IMAGES_DIRECTORY
print(path)
generator = ImageDataGenerator.flow_from_directory(
        CURRENT_DIRECTORY + IMAGES_DIRECTORY,
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
print(generator)

#input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
'''
def add_jpeg_decoding(input_width, input_height, input_depth, input_mean,
                      input_std):
    Adds operations that perform JPEG decoding and resizing to the graph..
    Args:
      input_width: Desired width of the image fed into the recognizer graph.
      input_height: Desired width of the image fed into the recognizer graph.
      input_depth: Desired channels of the image fed into the recognizer graph.
      input_mean: Pixel value that should be zero in the image for the graph.
      input_std: How much to divide the pixel values by before recognition.
    Returns:
      Tensors for the node to feed JPEG data into, and the output of the
        preprocessing steps.
    
    jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_shape = tf.stack([input_height, input_width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                             resize_shape_as_int)
    offset_image = tf.subtract(resized_image, input_mean)
    mul_image = tf.multiply(offset_image, 1.0 / input_std)
    return jpeg_data, mul_image        


    


'''