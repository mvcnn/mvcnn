import os
import tensorflow as tf

from abc import ABCMeta, abstractmethod
from functools import reduce

import re

TOWER_NAME = 'tower'


class Network(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    def _add_layer_summaries(self, model):
        '''
        Add layer summaries
        :param model: model object
        :return: merged layer summaries
        '''
        layer_summaries = []
        for layer in model:
            if layer.activation_summary is not None:
                layer_summaries.append(layer.activation_summary)
            if hasattr(layer, 'weight_summary'):
                layer_summaries.append(layer.weight_summary)
        layer_summaries_merged = tf.summary.merge(layer_summaries)
        return layer_summaries_merged
 

    def _add_grad_summaries(self, grads_and_vars):
        '''
        Add gradient summaries
        :param grads_and_vars: list of (gradient, variable) pairs as returned by compute_gradients()
        :return: merged gradient and variable summaries
        '''
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                grad_summaries.append(grad_hist_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)
        return grad_summaries_merged
    
    
    def _add_loss_summaries(self, total_loss, scope):
        '''
        Add loss summaries
        :param total_loss: total loss (cross entropy + weight decay)
        :param scope: current variable scope (string)
        :return: merged loss summaries
        '''
        losses = tf.get_collection('losses', scope)

        # Attach a scalar summary to all individual losses and the total loss; do the
        # same for the averaged version of the losses.
        loss_summaries = []
        for l in losses + [total_loss]:
            loss_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', l.op.name)
            loss_summaries.append(tf.summary.scalar(loss_name + ' (raw)', l))
        loss_summaries_merged = tf.summary.merge(loss_summaries)
        return loss_summaries_merged

    
    def accuracy(self, scores, labels):  # equivalent to average top-1 precision
        with tf.name_scope('accuracy'):
            labels = tf.cast(labels, tf.int64)
            correct_predictions = tf.equal(tf.argmax(scores, 1), labels)
            accuracy = 100 * tf.reduce_mean(tf.cast(correct_predictions, "float"))
            return accuracy

        
    def computational_cost(self, model):
        '''
        breakdown in terms of number of activations and weights for each layer
        :param model: model object
        :return: dictionary of layer name to number of activations, dictionary of layer name to number of weights
        '''
        # get dictionaries for memory/weights per layer
        activations = {}
        weights = {}
        for layer in model:
            # memory in bytes
            activations[layer.name] = reduce(lambda x, y: x * y, layer.shape[1:]) * 4
            if hasattr(layer, 'W'):
                weights[layer.name] = int(reduce(lambda x, y: x * y, layer.ksize))

        return activations, weights

    
    def loss(self, logits, labels, scope=None):
        '''

        :param logits: logits output by final layer (prior to softmax)
        :param labels: tensor containing labels for batch
        :param scope: variable scope
        :return: total loss (cross entropy + weight decay), loss summaries
        '''
        with tf.name_scope('loss'):
            # Calculate the average cross entropy loss across the batch.
            labels = tf.cast(labels, tf.int64)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels, name='cross_entropy_per_example')
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
            tf.add_to_collection('losses', cross_entropy_mean)

            # The total loss equals the cross entropy loss plus all of the weight
            # decay terms (L2 loss).
            total_loss = tf.add_n(tf.get_collection('losses', scope), name='total_loss')

            loss_summaries = self._add_loss_summaries(total_loss, scope)
            return total_loss, loss_summaries





