#!/bin/python
#coding:utf-8

class FeatureProcess:
    def __init__(self,data,\
                 sample=1.0, labels='label',\
                 normal_type=None, fillna_type=None,\
                 resample=False):
        self.data = data
        self.sample = sample
        self.labels = labels
        self.normal_type = normal_type
        self.fillna_type = fillna_type  
        self.resample = resample 

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

    def down_sample(self):
        if self.resample:
            pos_data = self.data[self.data[self.labels]==1].copy()
            neg_data = self.data[self.data[self.labels]==0].copy()
            pos_rate = len(pos_data)*1.0/len(neg_data)
            if pos_rate < 1:
                self.data =  pds.concat([pos_data, neg_data.sample(frac=pos_rate)],
                                ignore_index=True)
            else:
                pass
        else:
            pass 

    def pre_process(self):
        #drop label na
        self.data = self.data.sample(frac=self.sample)
        self.data = self.data.dropna(subset=[self.labels]).reset_index(drop=True)
        #balance sample
        self.down_sample()
        #bad cols check
        for i in self.data.columns:
            if self.data[i].count() < 2:
                print("cols:%s miss"%i)
                self.data.pop(i)
            elif len(self.data[i].unique()) < 2:
                print("cols:%s single value"%i)
                self.data.pop(i)
            elif self.data[i].dtype == 'str' or self.data[i].dtype == 'object':
                unique_arr = self.data[i].unique()
                key_map = {unique_arr[i]:i for i in range(len(unique_arr))}
                self.data[i] = self.data[i].map(lambda x:key_map[x])
            else:
                pass
        #normal
        self.normal_process()
        #fillna 
        self.fillna_process()
