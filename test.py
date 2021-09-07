#!/bin/python
#coding:utf-8
from .GeneralModel import GeneralModel
from .ModelParam import model_params
from .FeatureFilter import FeatSelect
import pandas as pds

if __name__=="__main__":
    
    df = pds.read_csv(file_name)
    #Step 1: Filter features from feats pool
    feat_model_params={
        'max_depth':6,
        'learning_rate':0.01
    }
    gmodel = GeneralModel(model='XGB_run',data=exp_data,
                         sample=0.2, normal_type='min_max',discrim=None,
                         fillna_type='drop',split_size=0.1,resample=False,
                         params=general_params['XGB_run'])
    gmodel.pre_process()
    data = gmodel.data
    fs = FeatSelect(data,top_k=30,params=feat_model_params)
    fs.collect()
    cols = fs.out_cols
    
    #Step 2: Machine Learning with filtered features, Xgboost/LR/RF/SVC...
    model_name = 'XGB_run'
    gmodel = GeneralModel(model=model_name,data=df[cols],
                        sample=1, normal_type='z_scale_sigma',discrim=None,
                        fillna_type='drop',split_size=0.1,resample=True,
                        params=model_params[model_name])
    model = gmodel.model_train()
