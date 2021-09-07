from sklearn.feature_selection import VarianceThreshold,chi2,SelectKBest,RFE,SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier 
from scipy.stats import pearsonr
from minepy import MINE
## feature selection
## 1.filter
#      1.1 remove by lower variance
class FeatSelect:
    def __init__(self,df,top_k,labels='label',params=None):
        self.data = df.copy()
        self.labels = labels
        self.k = top_k
        self.model = XGBClassifier(**params)
        self.score_dict = {}
        self.cols = self.data.columns.values
        for i in self.cols:
            self.score_dict[i] = 0
    #filter methods
    # low variance filter
    def filter_variance(self):
        var_row = self.data.describe().loc['std',:]
        var_list= [(self.cols[i],var_row[i]) for i in range(len(self.cols))]
        var_list = sorted(var_list,key=lambda x:x[1],reverse=True)
        print(var_list[:self.k])       
    def chiCheck(self):
        chi = SelectKBest(chi2,self.k)
        chi.fit_transform(self.data,self.data.label)
        for cols in chi.get_support(True):
            self.score_dict[self.cols[cols]] += 1
            print(self.cols[cols])
    def pearsonCheck(self):
        score_list=[]
        for i in range(len(self.cols)):
            score,_ = pearsonr(self.data[self.cols[i]], self.data.label)
            score_list.append((self.cols[i],abs(score)))
            self.score_dict[self.cols[i]] += abs(score)
        score_list = sorted(score_list,key=lambda x:x[1],reverse=True)
        print(score_list[:self.k])
    def MICscore(self):
        m = MINE()
        score_list=[]
        for i in range(len(self.cols)):
            m.compute_score(self.data[self.cols[i]], self.data.label)
            score_list.append((self.cols[i],abs(m.mic())))
            self.score_dict[self.cols[i]] += abs(m.mic())
        score_list = sorted(score_list,key=lambda x:x[1],reverse=True)
        print(score_list[:self.k])
    #model method
    def CrossVal(self):
        score_list=[]
        for i in range(len(self.cols)):
            X = self.data[self.cols[i]].values.reshape(-1,1)
            y = self.data.label.values
            score = cross_val_score(self.model, X, y, scoring="r2", cv=3)
            score_list.append((self.cols[i],format(np.mean(score),'.3f')))
            self.score_dict[self.cols[i]] += abs(np.round(np.mean(score),3))
        score_list = sorted(score_list,key=lambda x:x[1],reverse=True)
        print(score_list[:self.k])
    #Wrapper
    def RecursiveElim(self):
        rfe = RFE(self.model, n_features_to_select=self.k)
        rfe.fit_transform(self.data.drop('label',axis=1), self.data.label)
        for i in rfe.get_support(True):
            self.score_dict[self.cols[i]] += 1
            print(self.cols[i])
    #Embedded
    def LinearModelEm(self):
        lsvc = LinearSVC(C=0.01, penalty="l1",dual=False)
        lsvc.fit(self.data.drop('label',axis=1),self.data.label)
        model = SelectFromModel(lsvc, prefit=True,max_features=self.k)
        for i in model.get_support(True):
            self.score_dict[self.cols[i]] += 1
            print(self.cols[i])
    def collect(self):
        func_list = [
            'filter_variance',
            'chiCheck',
            'pearsonCheck',
            'MICscore',
            'CrossVal',
            'RecursiveElim',
            'LinearModelEm']
        for i in func_list:
            print(i)
            print("="*50)
            eval('self.'+i)()
            print("="*50)
        self.score_dict = sorted(self.score_dict.items(), key=lambda x:x[1], reverse=True)
        index = 0
        self.out_cols = []
        print("Total Weight")
        print("="*50)
        for item in self.score_dict:
            if index > self.k + 1:
                break
            else:
                print(item,'\n')
                self.out_cols.append(item[0])
            index = index + 1
