# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:09:54 2020

@author: deepa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from numpy.linalg import inv
import seaborn as sns
import random
import math



def normalizer(df):
    meanie=df.mean()
    stdd=df.std()
    x=(((df-meanie)/stdd))
    return x
url= 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
irisDataFrame = pd.read_csv(url, header=None)
irisDataFrame[4].replace('Iris-setosa','0',inplace=True)
irisDataFrame[4].replace('Iris-versicolor','1',inplace=True)
irisDataFrame[4].replace('Iris-virginica','1',inplace=True)
irisDataFrame[4]=irisDataFrame[4].astype('int32')
print(irisDataFrame.dtypes)
df=irisDataFrame[[0,1,4]]
df=df.astype('float64')
#df2 is df normalized
df2=normalizer(df[[0,1]])



y_train=np.array(df[4])
x_train=np.array(df2)
X_train=np.c_[np.ones(150),x_train]
x2_train=(df2[df[4]==0])
#x2_train=x2_train.drop(columns=4)
x2_train=np.array(x2_train)
y2_train=(df2[df[4]==1])
#y2_train=y2_train.drop(columns=4)
y2_train=np.array(y2_train)
plt.scatter(x2_train[:,0],x2_train[:,1],c='r')
plt.scatter(y2_train[:,0],y2_train[:,1],c='b')

#gradient Descent
Convg_criterion=0.000001
Max_Step=3000
Mu=5
alpa0=np.random.normal(0,1,3)
step=0
p=1/(1+np.exp(-1*np.dot(X_train,alpa0)))
grad=np.dot(p-y_train,X_train)
alpa1=alpa0-Mu*grad

while np.linalg.norm(alpa1-alpa0)>Convg_criterion and step<Max_Step:
    alpa0=alpa1
    p=1/(1+np.exp(-1*np.dot(X_train,alpa0)))
    grad=np.dot(p-y_train,X_train)
    alpa1=alpa0-Mu*grad
    step=step+1

pts_x=np.linspace(-2,3,num=200)
pts_y=(0.5-alpa0[0]-(alpa0[1]*pts_x))*(1/(alpa0[2]))
plt.scatter(x2_train[:,0],x2_train[:,1],c='r')
plt.scatter(y2_train[:,0],y2_train[:,1],c='b')
plt.plot(pts_x,pts_y,c='g')

n=150
X_test=X_train
C=np.zeros((2,2))
y_test_no=np.dot(X_test,alpa0)
y_test=np.sign(y_test_no-0.5)
i=0
while i<n:
    if y_test[i]==0:
        y_test[i]=np.random.randint(2,size=1)
        #print("Heloo")
    if y_test[i]==-1:
        y_test[i]=0
    i=i+1
y0=y_test[0:50]
y1=y_test[50:150]
C[1,0]=np.sum(y0)
C[0,0]=50-C[1,0]
C[1,1]=np.sum(y1)
C[0,1]=100-C[1,1]
acc=np.trace(C)/n
print('acc on training with 2 features and logistic regression is ', acc)

