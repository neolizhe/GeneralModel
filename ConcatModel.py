#!/bin/python
#coding:utf-8
import tensorflow as tf
import numpy as np
from Word2Vec import Word2Vec
from DeepModel import DeepModel
from tensorflow.python.tools import freeze_graph
import pyarrow.hdfs as hdfs
import os,re

class ConcatModel(DeepModel):
    def __init__(self, file_name, init_lr=0.001, batch_size=256, process='train', rm_dir=False, use_checkpoint=False):
        super().__init__(file_name, init_lr=init_lr, batch_size=batch_size, process=process, rm_dir=rm_dir, use_checkpoint=use_checkpoint)
        self.carpool_dir = './carpool_model_dir'
        self.embed_dir = './embed_model_dir'
    
    @classmethod
    def modify_raw_graph(cls, model_dir):
        try:
            open(model_dir + '/graph_new.pbtxt', 'r')
            print('exist!!')
            return 0
        except:
            print('generate new graph!')
        f_in = open(model_dir + '/graph.pbtxt', 'r')
        f_out = open(model_dir + '/graph_new.pbtxt', 'w')
        while(True):
            line = f_in.readline()
            if re.match('library', line):
                break
            else:
                _ = f_out.write(line)
        f_in.close()
        f_out.close()
        print("graph new generated!")
        return 0

    @classmethod
    def check_freeze_graph(cls, model_dir, out_name):
        try:
            open(model_dir + '/freeze_graph.pb', 'rb')
            pass
        except Exception as e:
            print("Generate freeze graph")
            freeze_graph.freeze_graph(input_graph = model_dir + '/graph_new.pbtxt',
                                    input_binary = False,
                                    input_saver = '',
                                    input_checkpoint = tf.train.latest_checkpoint(model_dir),
                                    output_node_names = out_name,
                                    restore_op_name = 'save/restore_all',
                                    filename_tensor_name = 'save/Const:0',
                                    output_graph = model_dir + '/freeze_graph.pb',
                                    clear_devices = True,
                                    initializer_nodes = ''
                                    )
            print("freeze graph generated")


    def load_DNN(self, basic_feat, dnn_path, name='loaded_dnn'):
        #self.check_freeze_graph(dnn_path)
        with tf.variable_scope(name):
            graph_buffer = tf.GraphDef.FromString(open(dnn_path + '/freeze_graph.pb', "rb").read())
            dnn_vec = tf.import_graph_def(graph_buffer, input_map={"deep_net/vec_in:0":basic_feat}, return_elements=["deep_net/vec_out:0"], name=name + '/')
            #for i,n in enumerate(graph_buffer.node):
            #    if n.name in ('IteratorV2', 'IteratorGetNext'):
            #        print(n)
        return dnn_vec[0]

    def load_Word2Vec(self, embed_vec, w2v_path, name='load_w2v'):
        #self.check_freeze_graph(w2v_path)
        with tf.variable_scope(name):
            graph_buffer = tf.GraphDef.FromString(open(w2v_path + '/freeze_graph.pb', "rb").read())
            w2v_vec = tf.import_graph_def(graph_buffer, input_map={"wordvec_layers/vec_in:0":embed_vec}, return_elements=["wordvec_layers/vec_out:0"], name=name + '/')
        return w2v_vec[0]

    def dense_layer(self, x, units=128, name='dense_layer'):
        with tf.variable_scope(name):
            initializer = tf.contrib.layers.variance_scaling_initializer()
            kernel = tf.get_variable('kernel', shape=[x.shape[1], units], regularizer=self.regular(), initializer=initializer)
            bias = tf.get_variable('bias', shape=[units,], initializer=tf.zeros_initializer())
            output = tf.nn.relu(x @ kernel + bias)
        return output
    
    def output_box(self, vec1, vec2, name='output_box1'):
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
            kernel = tf.get_variable('kernel', shape=[x.shape[1], 1], initializer=initializer)
            bias = tf.get_variable('bias', shape=[1,], initializer=tf.zeros_initializer())
            output = x @ kernel + bias
            #output = tf.nn.sigmoid(x @ kernel + bias)
        return output

    def output_layers(self, dnn_vec, w2v_vec, name="output_layers"):
        with tf.variable_scope(name):
            tmp_vec = dnn_vec
            for i in range(6):
                #bucket = self.dense_layer(w2v_vec, 64, name="bucket_%s"%i)
                #out = self.dense_layer(bucket, 16, name="bukect_out_%s"%i)
                tmp_vec = self.output_box(tmp_vec, w2v_vec, name="output_box_%s"%i)
                output = self.bottle_neck(tmp_vec, name="bottle_neck_%s"%i)
                if i == 0:
                    bucket_vec = output
                else:
                    bucket_vec = tf.concat([bucket_vec, output], axis=1)
        return bucket_vec

    def model_fn(self, features, labels, mode, params):
        features = features.get('features')
        basic_feat, embed_feat, wait_time, label = tf.split(features, [36, 25, 6, 1], axis=1)
        dnn_vec = self.load_DNN(basic_feat, self.carpool_dir)
        w2v_vec = self.load_Word2Vec(embed_feat, self.embed_dir)
        print('neo', dnn_vec.shape, w2v_vec.shape)
        multi_output = self.output_layers(dnn_vec, w2v_vec)
        predicts = wait_time * multi_output @ tf.ones([6, 1])
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
                'accuracy': tf.metrics.accuracy(label, tf.round(predicts)),
                'precision': tf.metrics.precision(label, tf.round(predicts)),
                'recall': tf.metrics.recall(label, tf.round(predicts)),
            }
        return tf.estimator.EstimatorSpec(mode, predicts, loss, train_op, metrics)
