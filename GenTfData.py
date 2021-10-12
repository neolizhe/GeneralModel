#!/bin/env tf1_15
#coding:utf-8
from FeatureProcess import FeatureProcess
import tensorflow as tf
import numpy as np
import pandas as pds
import argparse
import subprocess as sp
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from ModelParams import gentf_params

class GenTfData(FeatureProcess):
    def __init__(self, file_name, split_size=0.2):
        '''
        Generate tfrecord dataset for deep model,derived from FeatureProcess method.
        Input: $file_name.csv
        Ouput: Three files including $file_name_train.tfrecord, $file_name_valid.tfrecord, $file_name_valid.csv
        split_size: size for test dataset, train dataset size is 1 - split_size.
        work_path: origin data load path,
        hdfs_path: targeted tfrecord save path in hdfs
        '''
        FeatureProcess.__init__(self, data = 0, normal_type='z_scale_sigma', fillna_type='0')
        self.split_size = split_size
        self.file_name = file_name
        self.trainTf = self.file_name.split(".")[0]+"_train.tfrecord"
        self.validTf = self.file_name.split(".")[0]+"_valid.tfrecord"
        self.validCsv = self.file_name.split(".")[0]+"_valid.csv"
        self.work_path = gentf_params['work_path']
        self.hdfs_path = gentf_params['hdfs_path']

    def tfrecord_output(self, writer,df, chunk_iter, is_train='train'):
        df_res = df.copy()
        labels = df_res.pop('label')
        df_res['features']=df_res.apply(lambda x: x.values, axis=1)
        df_res['label']=labels
        with tqdm(range(len(df_res)),"chunkiter_%s_tfrecord_%s_processing..."%(chunk_iter,is_train)) as t:
            for i in t:
                feature=tf.train.Feature(float_list=tf.train.FloatList(value=df_res.loc[i,'features'].reshape(-1)))
                label=tf.train.Feature(float_list=tf.train.FloatList(value=[df_res.loc[i,'label']]))
                cols=tf.train.Features(feature={'features':feature,'label':label})
                example=tf.train.Example(features=cols)
                writer.write(example.SerializeToString())

    def toHDFS(self, file):
        outfile = self.hdfs_path + file
        try:
            sp.check_output(['hdfs','dfs','-rm', outfile])
        except Exception as e:
            print(e)
        sp.check_output(['hdfs','dfs','-put', outfile.split("/")[-1], outfile])
        sp.check_output(['rm', outfile.split("/")[-1]])
        print('file: %s trans done'%(outfile.split("/")[-1]))
    
    def protect_enumfeatures(self, enum_cols=[]):
        enum_df = self.data[enum_cols].copy()
        self.data = self.data.drop(labels = enum_cols, axis=1)
        self.pre_process()
        self.data = pds.concat([enum_df, self.data], axis=1).dropna()

    def cross_floatfeatures(self, float_cols=[]):
        float_df = self.data[float_cols].copy()
        self.data = self.data.drop(labels = float_cols, axis=1)
        self.data = pds.concat([float_df, self.data], axis=1)

    def collect(self):
        file_path = self.work_path + self.file_name
        #
        reader = pds.read_csv(file_path, iterator=True, chunksize=5*10**6)
        writer_1 = tf.python_io.TFRecordWriter(self.trainTf)
        writer_2 = tf.python_io.TFRecordWriter(self.validTf)
        valid_csv = []
        iters = 0
        for train_data in reader:
            self.data = train_data
            self.cross_floatfeatures(gentf_params['cross_features'])
            self.protect_enumfeatures(gentf_params['enum_features'])
            train_dataset, valid_dataset = train_test_split(self.data,test_size=self.split_size, stratify = self.data.label.values)
            train_dataset = train_dataset.reset_index(drop=True)
            valid_dataset = valid_dataset.reset_index(drop=True)
            valid_csv.append(valid_dataset)
            self.tfrecord_output(writer_1,train_dataset,iters,'train')
            self.tfrecord_output(writer_2,valid_dataset,iters,'valid')
            iters += 1
            print("tfrecord generate done")
        del train_data,train_dataset,valid_dataset
        writer_1.close()
        writer_2.close()
        pds.concat(valid_csv, ignore_index=True).to_csv(self.validCsv,index=False)
        del valid_csv
        self.toHDFS(self.trainTf)
        self.toHDFS(self.validTf)
        self.toHDFS(self.validCsv)
        print("Total Transfer Done")
