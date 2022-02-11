#!/bin/env tf1_15
#coding:utf-8
import argparse
import logging
import subprocess as sp
import numpy as np
import pandas as pds
import tensorflow as tf
import pyarrow.hdfs as hdfs
from sklearn.metrics import confusion_matrix,accuracy_score,auc,precision_score,roc_auc_score,recall_score
from datetime import datetime
from ModelParams import deep_params
from DeepModel import DeepModel


class Word2Vec(DeepModel):
    def __init__(self, file_name, init_lr = 1e-3, batch_size = 256, process='train', rm_dir=False, use_checkpoint=False):
        DeepModel.__init__(self, file_name, init_lr, batch_size, process, rm_dir,use_checkpoint)
        self.e_arr = 2*np.ones(25, dtype=np.int)
        self.e_arr[:3] = [300, 7, 24*12]
        self.e_sum = sum(self.e_arr)
        self.e_add_arr = [sum(self.e_arr[:i]) for i in range(len(self.e_arr))]

    def embedding_layer(self, x, name='embedding'):
        with tf.variable_scope(name):
            #print(x.shape, self.e_add_arr.shape,"neo shape")
            x = tf.cast(x, tf.int32)
            x = x + tf.constant(self.e_add_arr, dtype=tf.int32)
            initializer = tf.truncated_normal_initializer(stddev=1.0/(self.e_sum ** 0.5))
            kernel = tf.get_variable('kernel', (self.e_sum, self.struct['emvec_len']), initializer=initializer)
            output = tf.nn.embedding_lookup(kernel, x)
            output = tf.reshape(output, (-1, 25 * self.struct['emvec_len']))
        return output

    def linear_layer(self, x, units, name):
        with tf.variable_scope(name):
            initializer = tf.glorot_uniform_initializer()
            kernel = tf.get_variable('kernel',shape=[x.shape[1],units],initializer=initializer)
            bias = tf.get_variable('bias', shape=[units,],initializer=tf.zeros_initializer())
            output = x @ kernel + bias
        return output

    def softmax_layer(self, raw_logits, name='softmax'):
        with tf.variable_scope(name):
            logits = tf.nn.softmax(raw_logits, axis=1)
        return logits

    def wordvec_layers(self, features, mode, name='wordvec_layers'):
        with tf.variable_scope(name):
            #step1: embeding (?,N) --> (?,em_len*N)
            features = tf.add(features, 0, name='vec_in')
            em_vec = self.embedding_layer(features)
            #step2: nonlinear dense layers
            # 3 layers [128,128,64] all relu kernel with l2 regularize
            hidden_vec = self.deep_network(em_vec, mode)
            hidden_vec = tf.add(hidden_vec, 0, name='vec_out')
            #step3: hidden_vec to softmax output 6 buckets
            # 2 linear layers [32, 6]
            lo1 = self.linear_layer(hidden_vec, units=64, name='linear_out1')
            normal_out = self.linear_layer(lo1, units=6, name='linear_out2')
            #step4: softmax normalization (?,6)->(?,6)
            #normal_out = self.softmax_layer(lo2)
        return normal_out

    def model_fn(self, features, labels, mode, params):
        # network
        is_train = mode == tf.estimator.ModeKeys.TRAIN
        features = features.get('features')
        features, labels = tf.split(features,[25 ,6], axis=1)
        predicts = self.wordvec_layers(features, mode)
        train_op = metrics = loss = None
        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
            with tf.variable_scope('loss'):
                reg = tf.cast(tf.losses.get_regularization_loss(), tf.float32)
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = labels,logits = predicts)) + reg

        if mode == tf.estimator.ModeKeys.TRAIN:
            if self.use_ckp:
                logging.info('load checkpoint model from %s' % (params['checkpoint_path']))
                tf.train.init_from_checkpoint(params['checkpoint_path'], {'deep_net/': 'deep_net/'})
            with tf.variable_scope('optimizer'):
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                global_step = tf.train.get_or_create_global_step()
                warmup_done = tf.cast(global_step, tf.float32) / (params['lr_warmup_msteps'] * 1e6)
                warmup_lr = params['init_learning_rate'] * tf.minimum(warmup_done, 1.0)
                is_warmup = tf.cast(global_step, tf.float32) < (params['lr_warmup_msteps'] * 1e6)
                decay_lr = tf.train.polynomial_decay(
                    params['init_learning_rate'], global_step,
                    int(params['lr_decay_msteps'] * 1e6), params['end_learning_rate'])
                learning_rate = tf.where(is_warmup, warmup_lr, decay_lr)
                decay_rate = learning_rate * params['weight_decay_rate']
                tf.summary.scalar('learning_rate',learning_rate)
                optimizer = tf.contrib.opt.AdamGSOptimizer(learning_rate=learning_rate)
                #optimizer = tf.contrib.optimizer_v2.RMSPropOptimizer(learning_rate=params.init_learning_rate,decay=0.1,momentum=0.2)
                #optimizer = tf.contrib.opt.AdamWOptimizer(decay_rate, learning_rate)
                grads, variables = zip(*optimizer.compute_gradients(loss))
                grads, global_norm = tf.clip_by_global_norm(grads, 5)
                decay_vars = [v for v in variables if 'kernel:' in v.name]
                train_op = optimizer.apply_gradients(
                    zip(grads, variables), global_step)
    #, decay_var_list=decay_vars)
                train_op = tf.group([train_op, update_ops])
        if mode == tf.estimator.ModeKeys.EVAL:
            predicts = tf.nn.softmax(predicts, axis=1)
            metrics = {
                'mse':tf.metrics.mean_squared_error(labels, predicts),
                'mae':tf.metrics.mean_absolute_error(labels, predicts),
                'precision':tf.metrics.precision_at_thresholds(labels, predicts, [0.5]*6),
                'recall':tf.metrics.recall_at_thresholds(labels, predicts, [0.5]*6),
            }
        return tf.estimator.EstimatorSpec(mode, predicts, loss, train_op, metrics)
