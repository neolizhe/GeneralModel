# GeneralModel
Easy Mind ML Model Prototyping including LR/SVM/RF/XGBOOST...
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
