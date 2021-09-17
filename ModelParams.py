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

gentf_params = {
    'work_path':'~/data/',
    'hdfs_path':'~/hdfs/data/',
    'enum_feaures': ['city_id', 'week_day','call_time','is_onqueue'],
    'cross_features': ['carpool_rate_week','carpool_rate_avg_tol','avg_wait_dur'], #not use yet
}

deep_params = {
    'user_name':'neolizhe',
    'under_vs_order':0.5,
    'lr_warmup_msteps':0.05,
    'lr_decay_msteps':0.5,
    'end_learning_rate':1e-5,
    'weight_decay_rate':1,
    'log_msteps':0.01,
    'summary_msteps':0.01,
    'checkpoints_msteps':0.001,
    'train_msteps':0.2,
    'checkpoint_keep':10,
    'checkpoint_path':'./model_dir',
    'model_flag':'py',
    'throttle_secs':300,
    'model_dir':'./model_dir',
    'export_path':'./model_dir/export/',
    'test_ratio':0.2,
    'loss_weight':1,
    'structure':{
        #Input Dimension -- Enum Feature Dim, Numeric Feature Dim, label dim
        'e_num':4,
        'c_num':3,
        'n_num':34,
        'l_num':1,
        #Embedding Dim -- embedding input vector[size1, size2,,sizen], output vector size
        'e_size':[300, 7, 145, 2],
        'emvec_len':9,
        #Dense layer Param
        'dense_layers':3,
        'dense_units':[128, 64, 64],
        #Cross layer Param
        'cross_layers':3,
        #Dropout
        'drop_rate':0.5,
        #Regularizer
        'score':0.001,
    }
}
