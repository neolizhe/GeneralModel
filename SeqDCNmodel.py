#!/bin/python
#coding:utf-8
import tensorflow as tf
import numpy as np
from Word2Vec import Word2Vec
from DeepModel import DeepModel
# from tensorflow.python.tools import freeze_graph
import pyarrow.hdfs as hdfs
import os,re
class ConcatModel(DeepModel):
    def __init__(self, file_name, init_lr=0.001, batch_size=256, process='train', rm_dir=False, use_checkpoint=False):
        super().__init__(file_name, init_lr=init_lr, batch_size=batch_size, 
            process=process, rm_dir=rm_dir, use_checkpoint=use_checkpoint, use_normal=True)
        self.e_arr = 2*np.ones(25, dtype=np.int)
        self.e_arr[:3] = [300, 7, 24*12]
        self.e_sum = sum(self.e_arr)
        self.e_add_arr = [sum(self.e_arr[:i]) for i in range(len(self.e_arr))]
        self.normal_means = []
        self.normal_stds = []
        self.normal_file_path = './normal_params.txt'

    def load_normal_params(self):
        f = open(self.normal_file_path, 'r')
        for line in f.readlines():
            if line[:6] == "column":
                continue
            else:
                lines = line.strip().split(",")
                if len(lines) < 6:
                    continue
                else:
                    self.normal_means.append(float(lines[4]))
                    self.normal_stds.append(float(lines[5]))
        f.close()
        self.normal_means = self.normal_means[:36]
        self.normal_stds = self.normal_stds[:36]
        for x in range(36):
            if self.normal_stds[x] > -0.001 and self.normal_stds[x] < 0.001:
                self.normal_stds[x] = 1.0
            else:
                continue

    def dense_layer(self, x, units=128, name='dense_layer'):
        with tf.variable_scope(name):
            initializer = tf.contrib.layers.variance_scaling_initializer()
            kernel = tf.get_variable('kernel', shape=[x.shape[1], units], regularizer=self.regular(), initializer=initializer)
            bias = tf.get_variable('bias', shape=[units,], initializer=tf.zeros_initializer())
            output = tf.nn.relu(x @ kernel + bias)
        return output
    
    def deep_network(self, x, mode, name='deep_network'):
        ulist = [512,256,128]
        with tf.variable_scope(name):
            for i in range(len(ulist)):
                x = self.dense_layer(x, units = ulist[i],name="dense_layer_%s"%i)
                if i == 0:
                    x = tf.layers.batch_normalization(x, training= mode == tf.estimator.ModeKeys.TRAIN)
        return x
    
    def DNN(self, x, mode, name='dnn'):
        with tf.variable_scope(name):
            cross_output = self.cross_network(x, mode=mode)
            #Dense layers
            deep_output = self.deep_network(x, mode=mode)
            output = tf.concat([deep_output, cross_output], 1)
            ulist = [256, 128]
            for i in range(2):
                output = self.dense_layer(output, units = ulist[i], name="dense_layer_%s"%i)
        return output

    def Word2Vec(self, x, mode, name='w2v'):
        with tf.variable_scope(name):
            x = tf.cast(x, tf.int32)
            x = x + tf.constant(self.e_add_arr, dtype=tf.int32)
            initializer = tf.truncated_normal_initializer(stddev=1.0/(self.e_sum ** 0.5))
            kernel = tf.get_variable('kernel', (self.e_sum, 16), initializer=initializer)
            output = tf.nn.embedding_lookup(kernel, x)
            output = tf.reshape(output, (-1, 25 * 16))
            ulist = [512, 256, 128]
            for i in range(3):
                output = self.dense_layer(output, units = ulist[i], name="dense_layer_%s"%i)
        return output
    
    def output_box(self, vec1, vec2, mode, name='output_box1'):
        with tf.variable_scope(name):
            vec = tf.concat([vec1, vec2], axis=1)
            dense_vec = self.dense_layer(vec, 128, name='dense_1')
            cross_vec = self.normal_cross_layer(vec1, vec2)
            output = tf.concat([dense_vec, cross_vec], axis=1)
            output = self.dense_layer(output, 128, name='dense_2')
        return output

    def bottle_neck(self, x, name='bottle_neck'):
        with tf.variable_scope(name):
            initializer = tf.glorot_uniform_initializer() # sigmoid activate func use Xvaier Initializer
            kernel = tf.get_variable('kernel', shape=[x.shape[1],1], initializer=initializer)
            bias = tf.get_variable('bias', shape=[1,], initializer=tf.zeros_initializer())
            output = x @ kernel + bias
        return output

    def output_layers(self, dnn_vec, w2v_vec, mode, name="output_layers"):
        with tf.variable_scope(name):
            tmp_vec = dnn_vec
            for i in range(6):
                tmp_vec = self.output_box(tmp_vec, w2v_vec, mode=mode, name="output_box_%s"%i)
                output = self.bottle_neck(tmp_vec, name="bottle_neck_%s"%i)
                if i == 0:
                    bucket_vec = output
                else:
                    bucket_vec = tf.concat([bucket_vec, output], axis=1)
        return bucket_vec

    def model_fn(self, features, labels, mode, params):
        features, label = (features.get('features'), features.get('label'))
        basic_feat, embed_feat, wait_time = tf.split(features, [36, 25, 6], axis=1)
        basic_feat = (basic_feat - self.normal_means)/self.normal_stds/3
        basic_feat = tf.clip_by_value(basic_feat, -1, 1)
        dnn_vec = self.DNN(basic_feat, mode=mode)
        w2v_vec = self.Word2Vec(embed_feat, mode=mode)
        multi_output = self.output_layers(dnn_vec, w2v_vec, mode=mode)
        #predicts = wait_time * multi_output @ tf.ones([6, 1])
        product_id, _ = tf.split(basic_feat, [1, 35], axis=1)
        predicts = tf.concat([wait_time * multi_output, product_id], axis=1)
        predicts = self.dense_layer(predicts, 1, 'out_dense')
        predicts = tf.nn.sigmoid(predicts)
        train_op = metrics = loss = None
        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
            with tf.variable_scope('loss'):
                reg = tf.cast(tf.losses.get_regularization_loss(), tf.float32)
                loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(label, predicts, self.params['loss_weight']))

        if mode == tf.estimator.ModeKeys.TRAIN:
            if self.use_ckp:
                self.printLog('load checkpoint model from %s' % (params['checkpoint_path']))
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
                grads, variables = zip(*optimizer.compute_gradients(loss))
                grads, global_norm = tf.clip_by_global_norm(grads, 5)
                decay_vars = [v for v in variables if 'kernel:' in v.name]
                train_op = optimizer.apply_gradients(zip(grads, variables), global_step)
                train_op = tf.group([train_op, update_ops])
        if mode == tf.estimator.ModeKeys.EVAL:
            metrics = {
                'auc': tf.metrics.auc(label, predicts),
                'precision': tf.metrics.precision(label, tf.round(predicts)),
                'recall': tf.metrics.recall(label, tf.round(predicts)),
                'f1_score': tf.contrib.metrics.f1_score(label, tf.round(predicts))
            }
        return tf.estimator.EstimatorSpec(mode, predicts, loss, train_op, metrics)
