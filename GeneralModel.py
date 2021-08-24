#!/bin/python
#coding:utf-8

import xgboost
import pandas as pds
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,auc,precision_score,roc_auc_score,recall_score
import matplotlib.pyplot as plt
import sys,time
    
class GeneralModel:
    '''
        General ML Model Prototyping including LR/SVM/RF/XGBOOST...
        Processed by
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

        Params:

        model: 'LR_run', 'RF_run', 'SVC_run','XGB_run'
        data: pandas DataFrame
        labels: label column
        normal_type:'min_max','z_scale','z_scale_sigma'
        fillna_type:'0','mean','drop'
        split_size: test_size in train_test_split func
        resample: Use down-sample method if Positive counts << Negtive counts. Boolean True or False
        discrim: 'PCA','LDA','NCA' or None. Discriminant Method.
        params: dict for ML Model Params.
    '''
    def __init__(self,model,data,\
                 sample=1.0, labels='label',\
                 normal_type=None, fillna_type=None,\
                split_size=0.2, resample=False, \
                 discrim=None, params=None):
        self.data = data
        self.model = model
        self.sample = sample
        self.labels = labels
        self.normal_type = normal_type
        self.fillna_type = fillna_type
        self.split_size = split_size
        self.params = params
        self.resample = resample
        self.discrim = discrim
    
    def model_train(self):
        t0 = time.time()
        
        self.pre_process()
        t1 = time.time()
        print("data process consume %s s"%(t1-t0))
        
        input_args = self.dataset_gen()
        t2 = time.time()
        print("data generate consume %s s"%(t2-t1))
        
        model = eval('self.' + self.model)(input_args)
        t3 = time.time()
        print("model train consume %s s"%(t3-t2))
        self.modelpb = model
    # data process : normal / fillna or dropna
    def fillna_process(self):
        if self.fillna_type == '0':
            self.data = self.data.fillna(0)
        elif self.fillna_type == 'mean':
            self.data = self.data.fillna(self.data.mean())
        elif self.fillna_type == 'drop':
            self.data = self.data.dropna()
        else:
            pass
    def hard_minmax(self, x):
        if x>1:
            return 1
        elif x<-1:
            return -1
        else:
            return x
    def normal_process(self):
        #min_max to [0,1]
        #z_scale (x-miu)/sigma
        #z_scale (x-miu)/sigma/3 && hard_max
        if self.normal_type:
            tmp_labels = self.data.pop(self.labels)
            if self.normal_type == 'min_max':
                self.data = (self.data - self.data.min())/(self.data.max()-self.data.min())
            elif self.normal_type == 'z_scale':
                self.data = (self.data - self.data.mean())/self.data.std()
            elif self.normal_type == 'z_scale_sigma':
                self.data = (self.data - self.data.mean())/self.data.std()/3
                self.data = self.data.apply(lambda x: x.apply(self.hard_minmax), axis=1)
            else:
                pass
            self.data[self.labels] = tmp_labels
            del tmp_labels
        else:
            pass
        
    def pre_process(self):
        #drop label na
        self.data = self.data.sample(frac=self.sample)
        self.data = self.data.dropna(subset=[self.labels]).reset_index(drop=True)
        #bad cols check
        for i in self.data.columns:
            if self.data[i].count() < 2:
                print("cols:%s miss"%i)
                self.data.pop(i)
            elif len(self.data[i].unique()) < 2:
                print("cols:%s single value"%i)
                self.data.pop(i)
            else:
                pass
        #normal
        self.normal_process()
        #fillna 
        self.fillna_process()
    def down_sample(self, df):
        pos_data = df[df[self.labels]==1].copy()
        neg_data = df[df[self.labels]==0].copy()
        pos_rate = len(pos_data)*1.0/len(neg_data)
        if pos_rate < 1:
            return pds.concat([pos_data, neg_data.sample(frac=pos_rate)],
                             ignore_index=True)
        else:
            return df
        
    def discrim_train(self,x_train,y_train,x_test):
        if self.discrim == 'LDA':
            lda = LinearDiscriminantAnalysis(n_components = 2)
        elif self.discrim == 'PCA':
            lda = PCA(n_components=25, random_state=0)
        elif self.discrim == 'NCA':
            lda = NeighborhoodComponentsAnalysis(n_components=2, random_state=0)
        else:
            print("Err discrim type! return origin data")
            return x_train,y_train,x_test
        tmp = y_train[-3:].copy()
        y_train[-3:] = 2
        lda.fit(x_train, y_train)
        x_train = lda.transform(x_train)
        x_test = lda.transform(x_test)
        y_train[-3:] = tmp
        cols = ["x%s"%i for i in range(len(x_train[0]))]
        print("feature dims is %s after discrim"%len(cols))
        return x_train,y_train,x_test,cols
    
    def dataset_gen(self):
        train_data, test_data=train_test_split(self.data,test_size=self.split_size,\
                                              stratify = self.data[self.labels])
        #down sample
        if self.resample:
            train_data = self.down_sample(train_data)
        else:
            train_data = train_data.sample(frac=1.0)
        test_data = test_data.sample(frac=1.0)
        y_train = train_data.pop(self.labels).values
        x_train = train_data.values
        y_test = test_data.pop(self.labels).values
        x_test = test_data.values
        cols = train_data.columns.values
        #discrim
        if self.discrim:
            x_train,y_train,x_test,cols = self.discrim_train(x_train,y_train,x_test)

        if self.model == 'XGB_run':
            f_name = cols
            xtrain=xgboost.DMatrix(x_train,feature_names=f_name,missing=np.nan)
            xtrain.set_label(y_train)
            xtest=xgboost.DMatrix(x_test,feature_names=f_name,missing=np.nan)
            xtest.set_label(y_test)
            return (xtrain, xtest, x_test)
        else:
            return (x_train, y_train, x_test, y_test)
    #model SVC RF LR XGBOOST 
    def SVC_run(self, input_args):
        x_train, y_train, x_test, y_test = input_args
        if self.params:
            scvr=svm.SVC(**self.params)
        else:
            scvr=svm.SVC(C=1,kernel='rbf',gamma=8,decision_function_shape='ovo')
        scvr.fit(x_train, y_train)
        self.binary_evaluate(scvr,(x_test,y_test))
        return scvr
    
    def LR_run(self, input_args):
        x_train, y_train, x_test, y_test = input_args
        if self.params:
            lr = LogisticRegression(**self.params)
        else:
            lr = LogisticRegression()
        lr.fit(x_train, y_train)
        self.binary_evaluate(lr,(x_test,y_test))
        return lr
    
    def RF_run(self, input_args):
        x_train, y_train, x_test, y_test = input_args
        if self.params:
            rf = RandomForestClassifier(**self.params)
        else:
            rf = RandomForestClassifier()
        rf.fit(x_train, y_train)
        self.binary_evaluate(rf,(x_test,y_test))
        return rf
    
    def XGB_run(self, input_args):
        xtrain, xtest, x_test = input_args
        if self.params:
            pass
        else:
            self.params={
            'max_depth': max_depth,
            'subsample': 0.7,
            'objective': 'binary:logistic',
            'tree_method': 'exact',
            'eval_metric': ['auc'],
            'silent': 1,
            'scale_pos_weight': 2
            }
        watch_list = [(xtrain, 'train'), (xtest, 'eval')]
        res_dict={}
        n_es = self.params['num_boost_round']
        xgb=xgboost.train(self.params,xtrain,n_es,evals=watch_list,verbose_eval=False,evals_result=res_dict)
        fscore = xgb.get_fscore()
        fscore = sorted(fscore.items(), key=lambda x: x[1], reverse=True)
        print('#' * 60)
        print('fscore:')
        for k, v in fscore:
            print(k, v)
        print('#' * 60)
        sys.stdout.flush()
        print("over trainning")
        sys.stdout.flush()
        self.binary_evaluate(xgb, (xtest,x_test))
        train_value=res_dict['train'][self.params['eval_metric']]
        eval_value=res_dict['eval'][self.params['eval_metric']]
        res_pds=pds.DataFrame({'train':train_value,'eval':eval_value})
        res_pds.plot()
        return xgb
    
    def gen_label(self,x,thres):
        if x>=thres:
            return 1
        else:
            return 0

    def binary_evaluate(self, model, args):
        if self.model == 'XGB_run':
            xtest, x_test = args
            y_test_o=xtest.get_label()
            y_pred_o=model.predict(xtest)
        else:
            x_test, y_test_o = args
            y_pred_o = model.predict(x_test)
        cnt=len(y_pred_o)
        plt.hist(y_pred_o,bins=15,rwidth=0.5,cumulative=False,weights=[100/cnt]*cnt)
        plt.xlabel("Predict score ",fontsize=20)
        plt.ylabel("Percentage %",fontsize=20)
        print(np.min(y_pred_o),np.mean(y_pred_o),np.max(y_pred_o))
        pr,rc,ac,ps=0,0,0,0
        print("PostiveRatio:%.2f"%(sum([1 if x>0 else 0 for x in y_test_o])/len(y_test_o)),"TestDataSize:%s"%len(y_test_o),"TrainDataSize：%s"%(int(len(y_test_o)/0.2)))
        print("*"*50)
        for i in range(10):
            thres = i/10*np.max(y_pred_o)
            y_test=[self.gen_label(x,0.5) for x in y_test_o]
            y_pred=[self.gen_label(x,thres) for x in y_pred_o]
            pr=accuracy_score(y_test,y_pred)
            rc=recall_score(y_test,y_pred)
            ac=roc_auc_score(y_test,y_pred)
            ps=precision_score(y_test,y_pred)
            if rc>0.0 and ps>0.0:
                print(confusion_matrix(y_test,y_pred))
                print("Treshold: %.2f"%thres,'PostiveRatio: %.2f'% np.mean(y_pred))
                print("Accuracy:%.2f"%pr,"Recall:%.2f"%rc,"AUC:%.2f"%ac,"Precision:%.2f"%ps,"F1-score:%.2f"%(2*pr*rc/(pr+rc)))
                print("="*50)
    
    def output_model(self,save_path,city_param):
        f = open(save_path, 'w')
        f.write("area:%s\n"%city_param['city_id'])
        f.write("param {n_e} 1 {b_s} {m_depth} 0.0\n".\
                format(n_e=city_param['n_estimator'],b_s=city_param['decision_score'],m_depth=city_param['max_depth']))
        f.close()
        self.modelpb.dump_model(open(save_path,'a'))
        self.modelpb.save_model(save_path+"_r.bin")
        print("model saved in path: %s"%save_path)
        print("#"*20+"model head"+"#"*20)
        f = open(save_path, 'rt')
        print(f.read(100))
        f.close()
