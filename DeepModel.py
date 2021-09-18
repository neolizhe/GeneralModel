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
from GenTfData import GenTfData
    
class DeepModel(GenTfData):
    def __init__(self, file_name, init_lr = 1e-3, batch_size = 256, process='train', rm_dir=False, use_checkpoint=False):
        GenTfData.__init__(self, file_name)
        self.params = deep_params
        self.params['init_learning_rate'] = init_lr
        self.params['batch_size'] = batch_size
        self.process = process
        self.struct = self.params['structure']
        self.rm_dir = rm_dir
        self.use_ckp = use_checkpoint 
    #check if file exists
    def check_file(self):
        shellcmd = "hdfs dfs -test -e " + self.hdfs_path + self.trainTf + ";echo $?"
        shellout = sp.Popen(shellcmd, shell=True, stdout = sp.PIPE).communicate()
        if '1' not in str(shellout[0]):
            pass
        else:
            self.collect()
    
    def init_input_params(self):
        self.struct['prefetch_num'] = int(20480/self.params['batch_size'])
        self.struct['f_num'] = self.struct['e_num'] + self.struct['c_num'] + self.struct['n_num']
        self.struct['buffer_size'] = self.params['batch_size']*self.struct['prefetch_num']*4*self.struct['f_num']
        self.struct['shuffle_num'] = self.params['batch_size']*self.struct['prefetch_num']

    def input_fn(self, path, train=True):
        def f_parse(record):
            data = tf.parse_example(record, {
            'label': tf.FixedLenFeature([self.struct['l_num']], dtype=tf.float32),
            'features': tf.FixedLenFeature([self.struct['f_num']], dtype=tf.float32)})
            return data
        with tf.variable_scope('tf_reader'):
            filenames = hdfs.connect().ls(path)
            fileset = tf.data.Dataset.from_tensor_slices(filenames)
            if train: # shuffle file name
                fileset = fileset.shuffle(len(filenames)).repeat()
            dataset = tf.data.TFRecordDataset(fileset, buffer_size=self.struct['buffer_size'], num_parallel_reads=10)
            if train: # shuffle data
                dataset = dataset.shuffle(self.struct['shuffle_num'])
            return dataset.batch(self.params['batch_size']).map(f_parse).prefetch(self.struct['prefetch_num'])

    def serving_input_receiver_fn(self):
        inputs = {'features': tf.placeholder(shape=[None, self.struct['f_num']], dtype=tf.float32)}
        return tf.estimator.export.ServingInputReceiver(inputs, inputs)


    def embedding_layer(self, x, name='embedding'):
        with tf.variable_scope(name):
            e_size = sum(self.struct['e_size'])
            x = x + [sum(self.struct['e_size'][:i]) for i in range(len(self.struct['e_size']))]  # 类别特征从0开始依次往后排
            x = tf.cast(x, tf.int64)
            initializer = tf.truncated_normal_initializer(stddev=1.0/(e_size ** 0.5))
            kernel = tf.get_variable('kernel', (e_size, self.struct['emvec_len']), initializer=initializer)
            output = tf.nn.embedding_lookup(kernel, x)
            # output = tf.reshape(output, (-1, self.struct['e_num'] * self.struct['emvec_len']))
        return output

    def float_embedding(self, x, name='f_embedding'):
        with tf.variable_scope(name):
            useless_num = x.shape[1] % 3
            e_features,drop_features = tf.split(x, [x.shape[1] - useless_num, useless_num], 1)
            output = tf.reshape(e_features, (-1, 1, e_features.shape[1]))
        return output
        
    def regular(self):
        return tf.contrib.layers.l2_regularizer(self.struct['score'])

    def dense_layer(self, x, units=128, name='dense_layer'):
        assert x.shape.ndims == 2, '%s, %s' % (x.shape)
        with tf.variable_scope(name):
            initializer = tf.contrib.layers.variance_scaling_initializer(uniform=True)
            kernel = tf.get_variable('kernel', shape=[x.shape[1], units], regularizer=self.regular(), initializer=initializer)
            bias = tf.get_variable('bias', shape=[units,], initializer=tf.zeros_initializer())
            output = tf.nn.relu(x @ kernel + bias)
        return output

    def cross_product(self, tensor1, tensor2):
        tensor_list = []
        for i in range(tensor1.shape[1]):
            tensor0 = tf.gather(tensor1, tensor2.shape[1]*[i], axis = 1)
            tensor_list.append(tf.linalg.cross(tensor0, tensor2))
        return tf.concat(tensor_list, 1)

    def approxi_cross_operator(self, x, y):
        assert x.shape[2] % 3 == 0, "Error Embedding Dims: %s" % (x.shape, y.shape)
        split_num = x.shape[2]//3
        print(x.shape,y.shape,split_num)
        vec_array_x = tf.split(x, num_or_size_splits = split_num, axis = 2)
        vec_array_y = tf.split(y, num_or_size_splits = split_num, axis = 2)
        res_array = [self.cross_product(vec_array_x[i], vec_array_y[i]) for i in range(split_num)]
        return tf.concat(res_array, 2)

    def cross_product_layers(self, x):
        assert x.shape.ndims == 3, '%s' % x.shape
        x = tf.cast(x, tf.float32)
        with tf.variable_scope('cross_product_layers'):
            x0 = x
            for i in range(self.struct['cross_layers']):
                x = tf.concat([x, self.approxi_cross_operator(x, x0)], 1)
            initializer = tf.glorot_uniform_initializer()
            print(x.shape,"cross_neo")
            kernel = tf.get_variable('kernel', shape=[x.shape[2], 1], regularizer=self.regular(), initializer=initializer)
            bias = tf.get_variable('bias', shape=[x.shape[1], 1], initializer = tf.zeros_initializer())
            output =  x @ kernel + bias
            output = tf.reshape(output, (-1, x.shape[1]))
        return output

    def normal_cross_layer(self, x, x0, name='cross_layer'):
        assert x.shape.ndims == 2 and x0.shape.ndims == 2, '%s: %s, %s' % (x.shape, x0.shape)
        assert x.shape[1] == x0.shape[1], '%s: %s, %s' % (x.shape, x0.shape)
        with tf.variable_scope(name):
            initializer = tf.glorot_uniform_initializer()
            kernel = tf.get_variable('kernel', shape=[x.shape[1], 1], regularizer=self.regular(), initializer=initializer)
            bias = tf.get_variable('bias', shape=[x.shape[1],], initializer=tf.zeros_initializer())
            output = x0 * (x @ kernel) + bias + x
        return output

    def deep_network(self, x, name='deep_network'):
        #print(x.shape)
        assert x.shape.ndims == 2, '%s, %s' % (x.shape)
        with tf.variable_scope(name):
            for i in range(self.struct['dense_layers']):
                x = self.dense_layer(x, self.struct['dense_units'][i], 'dense_layer_%d' % i)
                if i % 2 == 0:
                    x = tf.layers.batch_normalization(x)
        return x

    def cross_network(self, x, name='cross_network'):
        with tf.variable_scope(name):
            x0 = x
            for i in range(self.struct['cross_layers']):
                x = self.normal_cross_layer(x, x0, 'cross_layer_%d' % i)
        return x

    def deep_net(self, x, mode, name='deep_net'):
        with tf.variable_scope(name):
            e_features, n_features = tf.split(x,[self.struct['e_num'], self.struct['c_num'] + self.struct['n_num']], 1)
            #Embedding layers
            embed_features = self.embedding_layer(e_features)
            cross_output = self.cross_product_layers(embed_features)
            #Dense layers
            deep_output = self.deep_network(n_features)
            output = tf.concat([deep_output, cross_output], 1)
            output = tf.layers.dropout(output, self.struct['drop_rate'], training= mode == tf.estimator.ModeKeys.TRAIN)
            # Output layer
            initializer = tf.glorot_uniform_initializer() # sigmoid activate func use Xvaier Initializer
            kernel = tf.get_variable('kernel', shape=[output.shape[1],1], initializer=initializer)
            bias = tf.get_variable('bias', shape=[1,], initializer=tf.zeros_initializer())
            output = tf.nn.sigmoid(output @ kernel + bias)
        return output

    def model_fn(self, features, labels, mode, params):
        # network
        is_train = mode == tf.estimator.ModeKeys.TRAIN
        features, labels = (features.get(key) for key in ['features', 'label'])  
        predicts = self.deep_net(features, mode)

        train_op = metrics = loss = None
        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
            with tf.variable_scope('loss'):
                reg = tf.cast(tf.losses.get_regularization_loss(), tf.float32)
                loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels, predicts, self.params['loss_weight']))
                loss = loss + reg

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
            metrics = {
                'auc': tf.metrics.auc(labels, predicts),
                'accuracy': tf.metrics.accuracy(labels, tf.round(predicts)),
                'precision': tf.metrics.precision(labels, tf.round(predicts)),
                'recall': tf.metrics.recall(labels, tf.round(predicts)),
            }
        return tf.estimator.EstimatorSpec(mode, predicts, loss, train_op, metrics)

    def gen_label(self, x, thres):
        if x>=thres:
            return 1
        else:
            return 0

    def infer_evaluate(self, predict_df):
        y_pred_o = predict_df.predicts.values
        y_test_o = predict_df.labels.values
        y_p = sorted(y_pred_o)
        res = {}
        for i in range(len(y_p)):
            key = np.round(y_p[i],2)
            if key in res.keys():
                res[key] += 1
            else:
                res[key] = 1
        for k,v in res.items():
            print("predict_score=%s"%k,"\t",int(v/len(y_p)*100)*"#","|")
        pr,rc,ac,ps=0,0,0,0
        print("PosRate:%.2f"%(sum([1 if x>0 else 0 for x in y_test_o])/len(y_test_o)),"TestSize:%s"%len(y_test_o))
        print("*"*50)
        for i in range(10):
            thres = i/10*np.max(y_pred_o)
            y_test=[self.gen_label(x,0.5) for x in y_test_o]
            y_pred=[self.gen_label(x,thres) for x in y_pred_o]
            pr=accuracy_score(y_test,y_pred)
            rc=recall_score(y_test,y_pred)
            ac=roc_auc_score(y_test,y_pred)
            ps=precision_score(y_test,y_pred)
            if rc>0.0 and ps>0.0:
                print(confusion_matrix(y_test,y_pred))
                print("Thres %.2f"%thres,'PosRate %.2f'% np.mean(y_pred))
                print("Acc:%.2f"%pr,"Recall:%.2f"%rc,"AUC:%.2f"%ac,"Precision:%.2f"%ps,"f1-score:%.2f"%(2*pr*rc/(pr+rc)))
                print("="*50)
    def remove_dir(self):
        if self.rm_dir:
            try:
                sp.check_output(['rm','-r',self.params['model_dir']])
            except Exception as e:
                print(e)
        else:
            pass

    def printLog(self, info_str):
        log_format = "%(asctime)s - %(levelname)s - %(message)s"
        logging.basicConfig(level=logging.INFO, format=log_format)
        infos = "="*(30 - len(info_str)//2) + info_str + "="*(30 - len(info_str)//2)
        logging.info(infos)

    def execute(self):   
        self.printLog("Step 1: Config")
        self.remove_dir()
        self.check_file()
        self.init_input_params()
        run_config = tf.estimator.RunConfig(
            model_dir = self.params['model_dir'],
            save_summary_steps = int(self.params['summary_msteps'] * 1e3),
            log_step_count_steps = int(self.params['log_msteps'] * 1e2),
            save_checkpoints_steps = int(self.params['checkpoints_msteps'] * 1e6),
            keep_checkpoint_max = self.params['checkpoint_keep']
        )
        
        estimator = tf.estimator.Estimator(model_fn=self.model_fn, config=run_config, params=self.params)
        best_exporter = tf.estimator.BestExporter(name='best_loss', 
                    serving_input_receiver_fn = self.serving_input_receiver_fn, exports_to_keep = 5)
        early_stopping = tf.estimator.experimental.stop_if_no_decrease_hook(estimator,'loss',5000,min_steps=10000)
        
        if self.process == "train":
            self.printLog("Train & Valid")
            train_spec = tf.estimator.TrainSpec(
                input_fn=lambda:self.input_fn(self.hdfs_path + self.trainTf, train=True),
                max_steps=int(self.params['train_msteps'] * 1e6), hooks = [early_stopping] )

            eval_spec = tf.estimator.EvalSpec(
                input_fn=lambda:self.input_fn(self.hdfs_path + self.validTf, train=False), 
                steps=None, exporters = best_exporter, throttle_secs=self.params['throttle_secs'])

            tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
            self.printLog("Finish Train & Valid")

        elif self.process == "valid":
            self.printLog("Valid")
            estimator.evaluate(
                input_fn=lambda:self.input_fn(self.hdfs_path + self.validTf, train=False),
                checkpoint_path = self.params['checkpoint_path'])
            self.printLog("Finish Valid Model")

        elif self.process == "infer":
            self.printLog("Start Infer")
            predicts = np.concatenate([p for p in estimator.predict(
            input_fn=lambda:self.input_fn(self.hdfs_path + self.validTf, train=False), checkpoint_path=self.params['checkpoint_path'])])
            fs = hdfs.connect()
            predict_df = pds.read_csv(fs.open(self.hdfs_path + self.validCsv), usecols=['label'])
            predict_df['predicts'] = predicts        
            predict_df['labels'] = predict_df.label
            predict_df = predict_df[['predicts','labels']]
            self.infer_evaluate(predict_df.sample(frac=self.params['test_ratio']))
            self.printLog("Finish Infer")
        
        elif self.process == "Export":
            self.printLog("Start Export")
            export = estimator.export_savedmodel(
                self.params['export_path'], self.serving_input_receiver_fn,
                checkpoint_path=self.params['checkpoint_path'], strip_default_attrs=True)
            self.printLog("Finish Export")
        else:
            self.printLog("Unknown Action")
