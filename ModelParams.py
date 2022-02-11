#!/bin/python
#coding:utf-8

ml_params={
    'LR_run':{
        'penalty':'l2',
        'tol':0.0001,
        'max_iter':100
    },
    'SVC_run':{
        'C':0.8,
        'kernel':'rbf',
        'tol':0.0001
    },
    'RF_run':{
        'n_estimators':500,
        'max_depth':6,
        'criterion':'entropy',
        'min_samples_split':2,
        'min_samples_leaf':2
    },
    'XGB_run':{
        'learning_rate':0.02,
        'max_depth':7,
        'num_boost_round':150,
        'objective': 'binary:logistic',
        'min_child_weight':1,
        'scale_pos_weight':1,
        'gamma':0,
        'lambda':0.5,
        'alpha':0.5,
        'eval_metric':'auc'
    }
}

deep_params = {
    'user_name':'neolizhe',
    'under_vs_order':0.5,
    'lr_warmup_msteps':0.01,
    'lr_decay_msteps':0.5,
    'end_learning_rate':1e-5,
    'weight_decay_rate':1,
    'log_msteps':0.01,
    'summary_msteps':0.01,
    'checkpoints_msteps':0.001,
    'train_msteps':1.0,
    'checkpoint_keep':10,
    'checkpoint_path':'./model_dir/model.ckpt-1000000',
    'model_flag':'py',
    'throttle_secs':300,
    'model_dir':'./model_dir',
    'export_path':'./model_dir/export/',
    'test_ratio':0.2,
    'loss_weight':2,
    'structure':{
        #Input Dimension -- Enum Feature Dim, Numeric Feature Dim, label dim
        'e_num':0,
        'c_num':0,
        'n_num':67,
        'l_num':1,
        #Embedding Dim -- embedding input vector[size1, size2,,sizen], output vector size
        'e_size':[4],
        'emvec_len':16,
        #Dense layer Param
        'dense_layers':3,
        'dense_units':[512,256,128],
        #Cross layer Param
        'cross_layers':3,
        #Dropout
        'drop_rate':0.5,
        #Regularizer
        'score':0.1,
    }
}
