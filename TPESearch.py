import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor,XGBClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,\
                    auc,precision_score,roc_auc_score,recall_score,mean_squared_error,\
                        mean_absolute_error,r2_score
def objwrapper(df):
    x,y = df.drop('label',axis=1).values, df.label.values
    x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.2)
    
    def objective(trial):
        params ={
            'learning_rate':trial.suggest_loguniform('learning_rate',1e-3,1e0),
            'max_depth':trial.suggest_int('max_depth',2,15),
            'n_estimators':trial.suggest_int('n_estimators',200,2000),
            'scale_pos_weight':trial.suggest_float('scale_pos_weight',1.0,10.0),
            'objective': 'binary:logistic',
            'min_child_weight':trial.suggest_int('min_child_weight',1,20),
            'gamma':trial.suggest_float('gamma',0.0,1.0),
            'reg_lambda':trial.suggest_float('reg_lambda',0.0,1.0),
            'reg_alpha':trial.suggest_float('reg_alpha',0.0,1.0),
            'subsample':trial.suggest_float('subsample',0.0,1.0),
            'colsample_bytree':trial.suggest_float('colsample_bytree',0.0,1.0),
            'colsample_bylevel':trial.suggest_float('colsample_bylevel',0.0,1.0),
        }
        xgbt=XGBRegressor(**params,verbosity=0)
        xgbt.fit(x_train,y_train)
        y_pred = xgbt.predict(x_test)
        y_pred = [1 if x >= 0.5 else 0 for x in y_pred]
        ps = precision_score(y_test,y_pred)
        rc = recall_score(y_test,y_pred)
        score = 2*ps*rc/(ps+rc+0.001)
        print('Precision:%s, Recall:%s, Score:%s'%(ps,rc,score))
        #score = sum([y_test[i] * (np.log(y_test[i] + 0.001) - np.log(y_pred[i] + 0.001)) for i in range(len(y_test))])/len(y_test)
        return score
    return objective

def TPESearch(df):
    study = optuna.create_study(sampler=TPESampler(),direction='maximize')
    objective = objwrapper(df)
    study.optimize(objective, n_trials=100,timeout=36000)
    print("Number of completed trials:%s"%len(study.trials))
    print("Best Trial:")
    trial = study.best_trial
    print("Best Score:%s"%(trial.value))
    print("Best Params:")
    for k,v in trial.params.items():
        print(k,v,'\n')
#     optuna.visualization.plot_param_importances(study)
#     optuna.visualization.plot_optimization_history(study)
    print("Optuna Search done!")
