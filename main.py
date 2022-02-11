#!/bin/env tf1_15
#coding:utf-8
from GeneralModel import GeneralModel
from ModelParams import ml_params
from DeepModel import DeepModel
from FeatureFilter import FeatSelect
from GenTfData import GenTfData
import pandas as pds
import sys

if __name__=="__main__":
    
    file_name = 'test.csv'
    df = pds.read_csv(file_name)
    process = sys.argv[1]
    
    if process == 'feature_filter':
    #Step 1: Filter features from feats pool
        feat_model_params={
            'max_depth':6,
            'learning_rate':0.01
        }
        fs = FeatSelect(df,top_k=30,params=feat_model_params)
        fs.collect()
        cols = fs.out_cols
        
    elif process == 'machine_learning':
    #Step 2: Machine Learning with filtered features, Xgboost/LR/RF/SVC...
        model_name = 'XGB_run'
        gmodel = GeneralModel(model=model_name,data=df,
                           sample=1, normal_type='z_scale_sigma',discrim=None,
                           fillna_type='drop',split_size=0.1,resample=True,
                           params=ml_params[model_name])
        model = gmodel.model_train()
        
    elif process == 'csv2tfrecord':
    #Step 3: Csv data to Tfrecord (preprocess for deepmodel)
        gtf = GenTfData(file_name=file_name)
        gtf.collect()
        
    elif process == 'deep_learning':
    #Step 4: Deep Model
        dmodel = DeepModel(file_name = file1, init_lr = 0.001, batch_size = 256, process='train')
        
    elif process == 'word2vec':
        dmodel = Word2Vec(file_name = file2, init_lr = 0.001, batch_size = 256, process='train')
        
    elif process == 'transfer_learning':
        #ConcatModel.modify_raw_graph('./carpool_model_dir')
        #ConcatModel.modify_raw_graph('./embed_model_dir')
        #ConcatModel.check_freeze_graph('./carpool_model_dir', 'deep_net/vec_in,deep_net/vec_out')
        #ConcatModel.check_freeze_graph('./embed_model_dir', 'wordvec_layers/vec_in,wordvec_layers/vec_out')
        dmodel = ConcatModel(file_name = file3, init_lr = 0.001, batch_size = 256, process='train')
    dmodel.execute()
    else:
        pass
