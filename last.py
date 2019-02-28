import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


train_data = pd.read_csv('train.csv')
test_data  = pd.read_csv('test.csv')





def Error_Using_CrossValidation(xgb_model,X_train,y_train):

     from sklearn.cross_validation import cross_val_score
     score =cross_val_score(xgb_model,X_train.values,y_train.values,cv=5,scoring='mean_absolute_error')
     score = -score
     m=np.max(score)
     return m 







def process(data):
    missing = [col for col in data.columns if data[col].isnull().any()]
    
  
    data[missing] = data[missing].fillna(0.0)
    
    obj = [col for col in data.columns if data[col].dtype==object]


    # for object data types
    data[obj] = data[obj].replace('yes',1)
    data[obj] = data[obj].replace('no',0)



    return data

def rate(x_train2,y_train2,y_test2,x_test2):
    xgb_model.fit(x_train2.values,y_train2.values)
    from sklearn.metrics import mean_absolute_error
    pred=xgb_model.predict(x_train2.values)
    m=mean_absolute_error(y_train2, pred)
    return m


def matrix(x_train2,y_train2,y_test2,x_test2):
    from sklearn.metrics import confusion_matrix
    xgb_model.fit(x_train2.values,y_train2.values)
    pred=xgb_model.predict(x_test2.values)
    cm = confusion_matrix(y_test2, pred)
    return cm



train_data = process(train_data)
test_data=process(test_data)









#Arrange The Data Set And Drop  'idhogar','Id' 



X_train=train_data.drop(['idhogar','Id','Target'], axis = 1)

y_train = train_data.iloc[:, 142]

X_test=test_data.drop(['idhogar','Id'], axis = 1)

    







# Run Algorithim
import xgboost as xgb
xgb_model = xgb.XGBClassifier(objective='multi:softmax', n_jobs=-1, num_class = 4,
                              random_state = 1021,silent=1,learning_rate=0.1,min_child_weight = 5,
                              max_depth=7,colsample_bytree= 0.85,subsample=0.85,
                              reg_alpha= 0.7,reg_lambda=0.2)

xgb_model.fit(X_train.values,y_train.values)
predict = xgb_model.predict(X_test.values)


sub = pd.DataFrame()

sub['Id'] = test_data['Id']

sub['Target'] = predict

sub.to_csv('submission.csv', index=False)

#ForConfusion MAtrix And Rate 
from sklearn.model_selection import train_test_split
x_train2, x_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=0.25, random_state = 0)













train_pred=xgb_model.predict(X_train.values)
 #For evaluating existing data(train)
train_data['pred_target'] = pd.Series(train_pred)
error=train_data[train_data['pred_target']!=train_data['Target']]
err=(len(error)/len(train_data))*100 #or err=(float(len(error))/len(df2))*100


def train_err(y_predicted, y_true):
    error=train_data[y_predicted!=y_true]
    err=(len(error)/len(train_data))*100 #or err=(float(len(error))/len(df2))*100
    print('False: ',err,'%\nCorrect: ',100-err,'%')


def error(y_predicted, y_true):
    
    rss = (y_predicted - y_true)**2
    mse = np.mean(rss)
    return mse


err1=train_err(train_data['pred_target'],train_data['Target'])
err2=error(train_data['pred_target'],train_data['Target'])
print('False: ',err2,'%\nCorrect: ',100-err2,'%')


m =matrix(x_train2,y_train2,y_test2,x_test2)
r =rate(x_train2,y_train2,y_test2,x_test2)

r_Cross= Error_Using_CrossValidation(xgb_model,X_train,y_train)       



