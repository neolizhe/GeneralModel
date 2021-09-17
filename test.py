#!/bin/env tf1_15
#coding:utf-8
from GeneralModel import GeneralModel
from ModelParams import ml_params
from DeepModel import DeepModel
from FeatureFilter import FeatSelect
import pandas as pds

if __name__=="__main__":
    
    file_name = 'test.csv'
    df = pds.read_csv(file_name)
    #Step 1: Filter features from feats pool
    feat_model_params={
        'max_depth':6,
        'learning_rate':0.01
    }
    fs = FeatSelect(df,top_k=30,params=feat_model_params)
    fs.collect()
    cols = fs.out_cols

    Step 2: Machine Learning with filtered features, Xgboost/LR/RF/SVC...
    model_name = 'XGB_run'
    gmodel = GeneralModel(model=model_name,data=df,
                       sample=1, normal_type='z_scale_sigma',discrim=None,
                       fillna_type='drop',split_size=0.1,resample=True,
                       params=ml_params[model_name])
    model = gmodel.model_train()

    #Step 3: Deep Model
    dmodel = DeepModel(file_name = 'origin_data.csv', batch_size=10240)
    dmodel.execute()
