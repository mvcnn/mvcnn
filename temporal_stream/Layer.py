import gzip
import os
import re
import sys
import math
from functools import reduce

import tensorflow as tf

# Create abstract Layer class with sub-classes (Conv, Max_Pool, etc.) from which a network will be built

from abc import ABCMeta, abstractmethod

TOWER_NAME = 'tower'
MOVING_AVERAGE_DECAY = 0.9
BN_EPSILON = 1e-3


class Layer(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    def _activation_summary(self, x):
        tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
        activations = tf.summary.histogram('/activations', x)
        sparsity = tf.summary.scalar('/sparsity', tf.nn.zero_fraction(x))
        self.activation_summary = tf.summary.merge([activations, sparsity])
        return self.activation_summary

    def _variable_on_cpu(self, name, shape, initializer, trainable=True):
        with tf.device('/cpu:0'):
            dtype = tf.float32
            var = tf.get_variable(name, shape=shape, initializer=initializer, dtype=dtype, trainable=trainable)
        return var

    def _variable_with_weight_decay(self, name, shape, stddev, wd):
        dtype = tf.float32
        var = self._variable_on_cpu(
            name,
            shape,
            tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def _visualize_weights(self, name, kernel, grid_X=8, pad=1):
        # TODO: add visualization of weights
        pass

# 2D/3D Conv layer
class Conv(Layer):
    def __init__(self, prev_layer, W_shape, stride_shape, name,
                 padding='SAME', BN=False, relu='relu', bias=None, wd=None, dropout_keep_prob=1.0,
                 visualize=False, type='2D', train=True):
        self.strides = [1] + stride_shape + [1]
        self.ksize = W_shape[:-1] + prev_layer.shape[-1:] + W_shape[-1:]
        self.name = name
        self.padding = padding
        self.BN = BN
        self.relu = relu
        self.bias = bias
        self.wd = wd
        self.dropout_keep_prob = dropout_keep_prob

        with tf.variable_scope(self.name) as scope:
            self.stddev = math.sqrt(2.0 / int(reduce(lambda x, y: x*y,self.ksize[:-1])))
            self.W = super(Conv, self)._variable_with_weight_decay('weights',
                                                                   shape=self.ksize,
                                                                   stddev=self.stddev,
                                                                   wd=self.wd)

            # Convolution for a given input and kernel
            if type is '2D':
                self.output = tf.nn.conv2d(prev_layer.output,
                                           self.W,
                                           self.strides,
                                           padding=self.padding)
            elif type is '3D':
                self.output = tf.nn.conv3d(prev_layer.output,
                                           self.W,
                                           self.strides,
                                           padding=self.padding)

            if self.bias != None:
                self.b = super(Conv, self)._variable_on_cpu('biases', [self.ksize[-1]],
                                                            tf.constant_initializer(self.bias))
                self.output = tf.nn.bias_add(self.output, self.b)

            if self.BN:
                pop_mean = super(Conv, self)._variable_on_cpu('pop_mean', [self.ksize[-1]],
                                                              tf.constant_initializer(0.0), trainable=False)
                pop_var = super(Conv, self)._variable_on_cpu('pop_var', [self.ksize[-1]],
                                                             tf.constant_initializer(1.0), trainable=False)

                scale = super(Conv, self)._variable_on_cpu('BN_scale', [self.ksize[-1]],
                                                           tf.constant_initializer(1.0))
                beta = super(Conv, self)._variable_on_cpu('BN_beta', [self.ksize[-1]],
                                                          tf.constant_initializer(0.0))

                if train:
                    axis = list(range(len(self.output.get_shape()) - 1))
                    batch_mean, batch_var = tf.nn.moments(self.output, axis)
                    train_mean = tf.assign(pop_mean,
                                           pop_mean * MOVING_AVERAGE_DECAY +
                                           batch_mean * (1 - MOVING_AVERAGE_DECAY))
                    train_var = tf.assign(pop_var,
                                          pop_var * MOVING_AVERAGE_DECAY +
                                          batch_var * (1 - MOVING_AVERAGE_DECAY))
                    with tf.control_dependencies([train_mean, train_var]):
                        self.output = tf.nn.batch_normalization(self.output, batch_mean, batch_var, beta,
                                                                scale, BN_EPSILON)
                else:
                    self.output = tf.nn.batch_normalization(self.output, pop_mean, pop_var, beta,
                                                            scale, BN_EPSILON)

            if self.relu == 'prelu':
                alphas = super(Conv, self)._variable_on_cpu('alpha', [self.ksize[-1]],
                                                            tf.constant_initializer(0.0))
                pos = tf.nn.relu(self.output)
                neg = tf.multiply(alphas, (self.output - tf.abs(self.output))) * 0.5
                self.output = pos + neg
            elif self.relu == 'relu':
                self.output = tf.nn.relu(self.output)

            self.shape = self.output.get_shape().as_list()
            self.activation_summary = super(Conv, self)._activation_summary(self.output)

            if self.dropout_keep_prob != 1.0:
                self.output = tf.nn.dropout(self.output, self.dropout_keep_prob)

# 2D/3D Max Pool layer
class Max_Pool(Layer):
    def __init__(self, prev_layer, window_shape, stride_shape, name, padding='VALID', type='2D'):
        self.ksize = [1] + window_shape + [1]
        self.strides = [1] + stride_shape + [1]
        self.name = name
        self.padding = padding
        self.activation_summary = None

        with tf.name_scope(self.name):
            if type is '2D':
                self.output = tf.nn.max_pool(prev_layer.output,
                                             ksize=self.ksize,
                                             strides=self.strides,
                                             padding=self.padding)
            elif type is '3D':
                self.output = tf.nn.max_pool3d(prev_layer.output,
                                               ksize=self.ksize,
                                               strides=self.strides,
                                               padding=self.padding)

            self.shape = self.output.get_shape().as_list()

# 2D/3D Average Pool layer
class Avg_Pool(Layer):
    def __init__(self, prev_layer, window_shape, stride_shape, name, padding='VALID', type='2D'):
        self.ksize = [1] + window_shape + [1]
        self.strides = [1] + stride_shape + [1]
        self.name = name
        self.padding = padding
        self.activation_summary = None

        with tf.name_scope(self.name):
            if type is '2D':
                self.output = tf.nn.avg_pool(prev_layer.output,
                                             ksize=self.ksize,
                                             strides=self.strides,
                                             padding=self.padding)
            elif type is '3D':
                self.output = tf.nn.avg_pool3d(prev_layer.output,
                                               ksize=self.ksize,
                                               strides=self.strides,
                                               padding=self.padding)

            self.shape = self.output.get_shape().as_list()

# Local Response Normalization layer
class LRN(Layer):
    def __init__(self, prev_layer, radius, alpha, beta, bias, name):
        self.radius = radius
        self.alpha = alpha
        self.beta = beta
        self.bias = bias
        self.name = name
        self.activation_summary = None

        # compute output shape
        self.shape = prev_layer.shape

        with tf.name_scope(self.name):
            self.output = tf.nn.local_response_normalization(prev_layer.output,
                                                             depth_radius=self.radius,
                                                             alpha=self.alpha,
                                                             beta=self.beta,
                                                             bias=self.bias)

# Fully Connected layer
class FC(Layer):
    def __init__(self, prev_layer, num_hidden, name, BN=False, relu='relu', bias=None,
                 wd=None, dropout_keep_prob=1.0, train=True):
        self.BN = BN
        self.relu = relu
        self.bias = bias
        self.wd = wd
        self.name = name
        self.dropout_keep_prob = dropout_keep_prob

        input_dim = int(reduce(lambda x, y: x * y, prev_layer.shape[1:]))
        self.ksize = [input_dim, num_hidden]

        with tf.variable_scope(self.name) as scope:
            self.stddev = 2.0 / input_dim
            self.W = super(FC, self)._variable_with_weight_decay('weights', shape=self.ksize,
                                                                 stddev=self.stddev, wd=self.wd)

            feed_in = tf.reshape(prev_layer.output, [-1, self.ksize[0]])
            self.output = tf.matmul(feed_in, self.W)
            if self.bias != None:
                self.b = super(FC, self)._variable_on_cpu('biases', [self.ksize[1]],
                                                          tf.constant_initializer(self.bias))
                self.output = self.output + self.b

            if self.BN:
                pop_mean = super(FC, self)._variable_on_cpu('pop_mean', [self.ksize[-1]],
                                                            tf.constant_initializer(0.0), trainable=False)
                pop_var = super(FC, self)._variable_on_cpu('pop_var', [self.ksize[-1]],
                                                           tf.constant_initializer(1.0), trainable=False)

                scale = super(FC, self)._variable_on_cpu('BN_scale', [self.ksize[-1]],
                                                         tf.constant_initializer(1.0))
                beta = super(FC, self)._variable_on_cpu('BN_beta', [self.ksize[-1]],
                                                        tf.constant_initializer(0.0))

                if train:
                    axis = list(range(len(self.output.get_shape()) - 1))
                    batch_mean, batch_var = tf.nn.moments(self.output, axis)
                    train_mean = tf.assign(pop_mean,
                                           pop_mean * MOVING_AVERAGE_DECAY +
                                           batch_mean * (1 - MOVING_AVERAGE_DECAY))
                    train_var = tf.assign(pop_var,
                                          pop_var * MOVING_AVERAGE_DECAY +
                                          batch_var * (1 - MOVING_AVERAGE_DECAY))
                    with tf.control_dependencies([train_mean, train_var]):
                        self.output = tf.nn.batch_normalization(self.output, batch_mean, batch_var, beta,
                                                                scale, BN_EPSILON)
                else:
                    self.output = tf.nn.batch_normalization(self.output, pop_mean, pop_var, beta,
                                                            scale, BN_EPSILON)

            if self.relu == 'prelu':
                alphas = super(FC, self)._variable_on_cpu('alpha', [self.ksize[-1]],
                                                          tf.constant_initializer(0.0))
                pos = tf.nn.relu(self.output)
                neg = tf.multiply(alphas, (self.output - tf.abs(self.output))) * 0.5
                self.output = pos + neg
            elif self.relu == 'relu':
                self.output = tf.nn.relu(self.output)

            self.shape = self.output.get_shape().as_list()
            self.activation_summary = super(FC, self)._activation_summary(self.output)

            if self.dropout_keep_prob != 1.0:
                self.output = tf.nn.dropout(self.output, self.dropout_keep_prob)


