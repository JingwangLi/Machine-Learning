# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 08:42:35 2017

Author: JingwangL
Email : 619620054@qq.com
Blog  : www.jingwangl.com

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import xgboost as xgb
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation, metrics
pd.set_option('display.width', 5000)

#读取数据
data1=pd.read_excel('Annex I.xlsx')
data2=pd.read_excel('Annex II translation.xlsx')
data3=pd.read_excel('Annex II.xlsx')
data4=pd.read_excel('Annex III.xlsx')

#获得所有诊断结果及其对应的样本数量
DiagnosisVar=data3.groupby([data3.Diagnosis]).size().sort_values(ascending=False)
#将样本数低于5的Diagnosis剔除，余下数据保存在Set中
DiagList=list(DiagnosisVar[DiagnosisVar>5].index)
Set=data3[data3.Diagnosis.isin(DiagList)]

#解决Set数据集不均衡问题,进行overSampling
maxNum=DiagnosisVar.max()
cnt=0
for i in DiagList:
    iSet=Set[Set.Diagnosis==i]
    n=maxNum-len(Set[Set.Diagnosis==i])
    choice=np.random.randint(0,len(iSet),n)
    for j in range(n):
        Set=Set.append(iSet.iloc[choice[j]],ignore_index=True)
        cnt+=1

##划分Train_set和Test_set
Set=Set[Set.Source=='Outpatient']
x=Set.iloc[:,1:].drop(['Source','Diagnosis'],axis=1)
y=Set.Diagnosis
x.loc[x.Sex=='male','Sex']=0
x.loc[x.Sex=='female','Sex']=1
cnt=0
for i in DiagList:
    y[y==i]=cnt
    cnt+=1
y=y.astype('int')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#选择算法进行训练及预测
#Logistic
#classifier=LogisticRegression()
#classifier.fit(x_train,y_train)
#predictions=classifier.predict(x_test)
#sum(predictions==y_test)/len(predictions)

#RandForest
#调参
#reasult=[]
#for i in range(1,200):
#    rf0 = RandomForestClassifier(oob_score=True, random_state=i)
#    rf0.fit(x_train,y_train)
#    y_predict = rf0.predict(x_test)
#    AccuracyRate=sum(y_predict==y_test)/len(y_test)
#    reasult.append([i,AccuracyRate])
#
#reasult=np.array(reasult)

#Best_random_state=int(reasult[reasult[:,1].argmax()][0])
#Best_random_state==36
rf0 = RandomForestClassifier(oob_score=True, random_state=36)
rf0.fit(x_train,y_train)
y_predict = rf0.predict(x_test)
AccuracyRate0=sum(y_predict==y_test)/len(y_test)
print(AccuracyRate)
#  0.961656356239


#GBDT
#rf0 = GradientBoostingClassifier(random_state=Best_random_state)
#rf0.fit(x_train,y_train)
#y_predict = rf0.predict(x_test)
#AccuracyRate=sum(y_predict==y_test)/len(y_test)
#print(AccuracyRate)
# 0.82071071626873959

##xgboost
#from sklearn import preprocessing
#for f in x_train.columns:
#    if x_train[f].dtype=='object':
#        lbl = preprocessing.LabelEncoder()
#        lbl.fit(list(x_train[f].values))
#        x_train[f] = lbl.transform(list(x_train[f].values))
#for f in x_test.columns:
#    if x_test[f].dtype=='object':
#        lbl = preprocessing.LabelEncoder()
#        lbl.fit(list(x_test[f].values))
#        x_test[f] = lbl.transform(list(x_test[f].values))
#
#x_train.fillna((-999), inplace=True)
#x_test.fillna((-999), inplace=True)
#
#x_train=np.array(x_train)
#x_test=np.array(x_test)
#x_train = x_train.astype(int)
#x_test = x_test.astype(int)
#rf0 = xgb.XGBClassifier(
# learning_rate =0.1,
# n_estimators=1000,
# max_depth=5,
# gamma=0,
# subsample=0.8,
# colsample_bytree=0.8,
# objective= 'multi:softmax',
# nthread=4,
# seed=27)
#rf0.fit(x_train,y_train)
#y_predict = rf0.predict(x_test)
#AccuracyRate=sum(y_predict==y_test)/len(y_test)
#print(AccuracyRate)

#a=pd.DataFrame(columns=['predict','test'])
#a.predict=y_predict
#a.test=list(y_test)

#对新数据进行预测
x_new=data4.iloc[:,1:].drop(['Source'],axis=1)
x_new.loc[x_new.Sex=='male','Sex']=0
x_new.loc[x_new.Sex=='female','Sex']=1
y_predict = rf0.predict(x_new)
#输出预测结果
for i in y_predict:
    print(DiagList[i])
