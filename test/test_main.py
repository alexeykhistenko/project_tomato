# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import mnist

# Basic model parameters as external flags.
FLAGS = None

# Constants used for dealing with the files, matches convert_to_records.
TRAIN_FILE = 'train-00000-of-00001.tfrecords'
VALIDATION_FILE = 'validation-00000-of-00001.tfrecords'

IMAGE_SIZE = 100
NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1100
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 300

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image/classtext':tf.FixedLenFeature([], tf.string),
            'image/fielname':tf.FixedLenFeature([], tf.string),
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/label': tf.FixedLenFeature([], tf.int64),
        })
#    print(features)
    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image/encoded'], tf.uint8)
    image.set_shape([mnist.IMAGE_PIXELS])
    
    # OPTIONAL: Could reshape into a 28x28 image and apply distortions
    # here.  Since we are not applying any distortions in this
    # example, and the next step expects the image to be flattened
    # into a vector, we don't bother.
    
    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5 ###
#    print(image)
    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['image/label'], tf.int32)
    
    filename = tf.cast(features['image/fielname'], tf.string)
    
#    print(filename)
#    tf.Print(filename, [filename])
#    filename = tf.Print(filename, [filename])
#    v = filename.eval()
#    print(v)
    
    return image, label, filename


def inputs(train, batch_size, num_epochs):
    """Reads input data num_epochs times.
    Args:
      train: Selects between the training (True) and validation (False) data.
      batch_size: Number of examples per returned batch.
      num_epochs: Number of times to read the input data, or 0/None to
         train forever.
    Returns:
      A tuple (images, labels), where:
      * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
        in the range [-0.5, 0.5].
      * labels is an int32 tensor with shape [batch_size] with the true label,
        a number in the range [0, mnist.NUM_CLASSES).
      Note that an tf.train.QueueRunner is added to the graph, which
      must be run using e.g. tf.train.start_queue_runners().
    """
    if not num_epochs: num_epochs = None
    filename = os.path.join(FLAGS.train_dir,
                            TRAIN_FILE if train else VALIDATION_FILE)
#    print(filename)
    
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            [filename], num_epochs=num_epochs)
#        print(filename_queue)
        
        # Even when reading in multiple threads, share the filename
        # queue.
        image, label, filename= read_and_decode(filename_queue)
        
        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        images, sparse_labels = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size, num_threads=2,
            capacity=1000 + 3 * batch_size,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=1000)
        
        return images, sparse_labels, filename


#===============================================================================

def cnn_fn(features, labels, mode):
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    print(features[0])
    input_layer = tf.reshape(features[0], [-1, 100, 100, 1])
    print('input_layer =', input_layer)
    
#    input_layer = tf.reshape(features["x"], [-1, 100, 100, 1])
    
        # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    print('conv1 =', conv1)
    
    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    print('pool1 =', pool1)
    
    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    print('conv2 =', conv2)
    
    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    print('pool2 =', pool2)
    
    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 7, 7, 64]
    # Output Tensor Shape: [batch_size, 7 * 7 * 64]
    pool2_flat = tf.reshape(pool2, [-1, 10 * 10 * 64])
    print('pool2_flat =', pool2_flat)
    
    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    print('dense =', dense)
    
    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
#    print('conv1 =', conv1)
    
    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(inputs=dropout, units=10)
    print('logits =', logits)

#===============================================================================

def main(unused_argv):
    # Load training and eval data
#    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
#    train_data = np.array(os.)
#    train_data = mnist.train.images  # Returns np.array
#    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
#    eval_data = mnist.test.images  # Returns np.array
#    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    
    
    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_fn, model_dir="/tmp/mnist_convnet_model")
    
    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)
    
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])
    
    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


#def main(_):   
#    cnn_fn(inputs(True, 100, None), inputs(True, 100, None), tf.estimator.ModeKeys.TRAIN)
#    print(inputs(True, 100, None))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=2,
        help='Number of epochs to run trainer.'
    )
    '''
    parser.add_argument(
        '--hidden1',
        type=int,
        default=128,
        help='Number of units in hidden layer 1.'
    )
    parser.add_argument(
        '--hidden2',
        type=int,
        default=32,
        help='Number of units in hidden layer 2.'
    )
    '''
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size.'
    )
    parser.add_argument(
        '--train_dir',
        type=str,
        default='C:/Users/I347798/git/project_tomato/test/OUTPUT_DIR',
        help='Directory with the training data.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    