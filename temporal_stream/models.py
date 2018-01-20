import os
import tensorflow as tf
from functools import reduce

from Layer import *
import math

from Network import Network


class Temporal_Stream(Network):
    def __init__(self, data, num_classes, wd_coeff=0.0, dropout_keep_prob=1.0, reuse=False, train=True, scope=None):

        # 3D model
        with tf.variable_scope("CIFAR_model", reuse=reuse):
            # reshape data form 3D to 4D input, by splitting the final dimension into  dx and dy channels
            data.output = tf.stack(tf.split(axis=3, num_or_size_splits=data.shape[-1]/2, value=data.output),
                                   axis=-1)

            # permute to [batch size, depth, height, width, channels]
            perm = [0, 4, 1, 2, 3]
            data.output = tf.transpose(data.output, perm=perm)
            data.shape = data.output.get_shape().as_list()

            conv1 = Conv(data, [3, 3, 3, 64], [2, 1, 1], bias=0.0, relu='prelu',
                         padding='SAME', name='conv1', type='3D', BN=False, train=train)
            pool1 = Max_Pool(conv1, [2, 2, 2], [2, 2, 2], padding='SAME', name='pool1', type='3D')
            conv2 = Conv(pool1, [3, 3, 3, 128], [2, 1, 1], bias=0.0, relu='prelu',
                         padding='SAME', name='conv2', type='3D', BN=False, train=train)
            pool2 = Max_Pool(conv2, [2, 2, 2], [2, 2, 2], padding='SAME', name='pool2', type='3D')
            conv3 = Conv(pool2, [2, 2, 2, 256], [1, 1, 1], bias=0.0, relu='prelu',
                         padding='SAME', name='conv3', type='3D', BN=False, train=train)
            conv4 = Conv(conv3, [2, 2, 2, 256], [1, 1, 1], bias=0.0, relu='prelu',
                         padding='SAME', name='conv4', type='3D', BN=False, train=train)
            conv5 = Conv(conv4, [2, 2, 2, 256], [1, 1, 1], bias=0.0, relu='prelu',
                         padding='SAME', name='conv5', type='3D', BN=False, train=train)
            pool3 = Max_Pool(conv5, [2, 2, 2], [2, 2, 2], padding='SAME', name='pool3', type='3D')
            fc1 = FC(pool3, 2048, relu='prelu', bias=0.0, BN=False,
                     wd=wd_coeff, dropout_keep_prob=dropout_keep_prob,
                     name='fc1', train=train)
            fc2 = FC(fc1, 2048, relu='prelu', bias=0.0, BN=False,
                     wd=wd_coeff, dropout_keep_prob=dropout_keep_prob,
                     name='fc2', train=train)
            fc3 = FC(fc2, num_classes, relu=None, BN=False, bias=0.0, name='fc3')

            model = [conv1, pool1, conv2, pool2, conv3, conv4, conv5, pool3, fc1, fc2, fc3]

        # add accuracy, accuracy summary
        logits = tf.squeeze(fc3.output)
        self.scores = tf.nn.softmax(logits)
        self.accuracy = super(Temporal_Stream, self).accuracy(self.scores, data.labels)
        self.activations, self.weights = super(Temporal_Stream, self).computational_cost(model)

        summaries = [data.image_summary]

        if train:
            #add layer summaries (activations plus weight visualizations)
            layer_summaries = super(Temporal_Stream, self)._add_layer_summaries(model)
            #add loss, loss summaries
            self.total_loss, loss_summaries = super(Temporal_Stream, self).loss(logits, data.labels, scope=scope)
            summaries.append([loss_summaries, layer_summaries])

        # merge model summaries
        self.merged_summaries = tf.summary.merge(summaries)


