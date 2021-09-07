# GeneralModel
Easy Operation Machine Learning in one line , including LR/SVM/RF/XGBOOST...
# Process
    1. Normalization. 
        Method: Min-Max,Z_scale,Z_scale_6sigma
    2. FillNan. 
        Method: Zero,Mean,Drop
    3. Discriminant. 
        Method: LDA/PCA/NCA method alternatively.
    4.ML Model.
        Method: LR/RF/XGBOOST/SVM and so on
    5.Evaluate.
        Method: Accuracy/Recall/Precision/F1-score/Confusion-Matrix/Hist_Plot
# Params
    model: 'LR_run', 'RF_run', 'SVC_run','XGB_run'
    data: pandas DataFrame
    labels: label column
    normal_type:'min_max','z_scale','z_scale_sigma'
    fillna_type:'0','mean','drop'
    split_size: test_size in train_test_split func
    resample: Use down-sample method if Positive counts << Negtive counts. Boolean True or False
    discrim: 'PCA','LDA','NCA' or None. Discriminant Method.
    params: dict for ML Model Params.
# Environment
    numpy>=1.18.5
    pandas>=1.0.5
    matplotlib>=3.2.2
    xgboost>=1.1.1
    sklearn>=0.23.1
# Demo
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
