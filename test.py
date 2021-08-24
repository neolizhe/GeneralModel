#!/bin/python
#coding:utf-8
from .GeneralModel import GeneralModel
from .ModelParam import model_params
import pandas as pds

if __name__=="__main__":
    
    df = pds.read_csv(file_name)
    model_name = 'XGB_run'
    gmodel = GeneralModel(model=model_name,data=df,
                        sample=1, normal_type='z_scale_sigma',discrim=None,
                        fillna_type='drop',split_size=0.1,resample=True,
                        params=model_params[model_name])
    model = gmodel.model_train()
