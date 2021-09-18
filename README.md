# GeneralModel
Easy Operating Feature Selection & Machine Learning & Deep Learning Model, including LR/SVM/RF/XGBOOST... DNN/DCN
# Structure
## BASE: FeatureProcess 
--- Process Input data including Normalization & Fillna & Discriminant\
    1. Normalization. \
        Method: Min-Max,Z_scale,Z_scale_6sigma\
    2. FillNan. \
        Method: Zero,Mean,Drop\
    3. Discriminant. \
        Method: LDA/PCA/NCA method alternatively.\
## FeatureFilter (Derived from FeatureProcess)
--- Select TOP K Best Features including varies methods.\
    1.Filter by Variance.\
        Drop columns with variance lower than threshold.\
    2.chiCheck\
        TOP K features sorted by chi(features, label)\
    3.Pearson Check\
        TOP K features sorted by pearsonr(features, label)\
    4.MIC Score\
        Method derived from  sklearn MINE()\
    5.Model Based\
        Use ML Model like LR/Xgboost to sort features by their importance.\
## GeneralModel (Derived from FeatureProcess)
--- Machine Learning Model including LR/SVM/RF/XGBOOST
### Params
    model: 'LR_run', 'RF_run', 'SVC_run','XGB_run'
    data: pandas DataFrame
    labels: label column
    normal_type:'min_max','z_scale','z_scale_sigma'
    fillna_type:'0','mean','drop'
    split_size: test_size in train_test_split func
    resample: Use down-sample method if Positive counts << Negtive counts. Boolean True or False
    discrim: 'PCA','LDA','NCA' or None. Discriminant Method.
    params: dict for ML Model Params.
## GenTfData (Derived from FeatureProcess)
--- Transfer dataset.csv to three .tfrecord files for tensorflow deep model: trainset.tfrecord/ validset.tfrecord/ validset.csv.\
    Input: $file_name.csv\
    Ouput: Three files including $file_name_train.tfrecord, $file_name_valid.tfrecord, $file_name_valid.csv\
    split_size: size for test dataset, train dataset size is 1 - split_size.
    work_path: origin data load path\
    hdfs_path: targeted tfrecord save path in hdfs\
## DeepModel (Derived from GenTfData)
--- Classic Deep Learning Method including DNN/ DCN.
### Params
    1.structure: Model Structre Params.
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
        'score':0.001
     2.Train Optimizer Params (msteps stands for million steps)
        'lr_warmup_msteps':0.05, #learning rate warmup * million steps
        'lr_decay_msteps':0.5, #learning rate decay
        'end_learning_rate':1e-5,
        'weight_decay_rate':1,
        'log_msteps':0.01, # log * million steps
        'summary_msteps':0.01,
        'checkpoints_msteps':0.001,
        'train_msteps':0.2, # train steps * millon
# Environment
    numpy>=1.18.5
    pandas>=1.0.5
    matplotlib>=3.2.2
    xgboost>=1.1.1
    sklearn>=0.23.1
# Demo
    from GeneralModel import GeneralModel
    from ModelParams import ml_params
    from DeepModel import DeepModel
    from FeatureFilter import FeatSelect
    import pandas as pds

    if __name__=="__main__":

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
        
        #Step 3: Deep Model
        dmodel = DeepModel(file_name = 'origin_data.csv', batch_size=10240)
        dmodel.execute()
