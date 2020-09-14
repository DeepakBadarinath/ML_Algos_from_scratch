# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:42:29 2020

@author: deepa
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 08:17:21 2020

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

def normalizer_numpy(a):
    meanie=np.mean(a,axis=0)
    stdd=np.std(a,axis=0)
    b=(a-meanie)/stdd
    return (b,meanie,stdd)

def Kernel(x,y):
    return np.dot(x,y)

def OneStepAlgo(i,j,X,y,f,beta,b,C):
    delta=y[i]*((f[j]-y[j])-(f[i]-y[i]))
    s=y[i]*y[j]
    chi=Kernel(X[i,:],X[i,:])+Kernel(X[j,:],X[j,:])-2*Kernel(X[i,:],X[j,:])
    gamma=s*beta[i]+beta[j]
    if s==1:
        L=max(0,gamma-C)
        H=min(gamma,C)
    else:
        L=max(0,-1*gamma)
        H=min(C,C-gamma)
    if chi>0:
        beta_i_new=min(max(beta[i]+(delta/chi),L),H)
    elif delta>0:
        beta_i_new=L
    else:
        beta_i_new=H
    beta_j_new=gamma-s*beta_i_new
    f_new=f+(beta_j_new-beta[j])*y[j]*Kernel(X[j,:],np.transpose(X))+(beta_i_new-beta[i])*y[i]*Kernel(X[i,:],np.transpose(X))
    b_new=b-0.5*(f_new[i]-y[i]+f_new[j]-y[j])
    return (beta_i_new,beta_j_new,f_new,b_new)


def Distinct_Two_Random(n):
    i=np.random.randint(0,n)
    j=np.random.randint(0,n-1)
    if j>=i:
        j=j+1
    return (i,j)

#Data Collection and Scaling with initial plot
x_0=np.random.exponential(scale=0.25,size=(20,2))
x_1=np.random.exponential(scale=2,size=(20,2))
x_train=np.r_[x_0,x_1]
(x_train,meanie,stdd)=normalizer_numpy(x_train)
x_0=x_train[0:20,:]
x_1=x_train[20:40,:]
X_train=x_train
X_train_Reg=np.c_[np.ones(40),x_train]
plt.scatter(x_0[:,0],x_0[:,1],c='r')
plt.scatter(x_1[:,0],x_1[:,1],c='b')

#Construction of training labels
zo=np.zeros(20)-1
oz=np.ones(20)
oz=oz.reshape((20,1))
zo=zo.reshape((20,1))
y_train=np.r_[zo,oz]


#Gathering validation data
x_0_test=np.random.exponential(scale=0.25,size=(500,2))
x_1_test=np.random.exponential(scale=2,size=(500,2))
x_test=np.r_[x_0_test,x_1_test]
y_actual=np.r_[np.zeros(500)-1,np.ones(500)]

#Scaling validation data
x_test=(x_test-meanie)/stdd
X_test=x_test
X_test_Reg=np.c_[np.ones(1000),X_test]

#SVM initialisations and parameter setting
beta=np.zeros(40)
b=0
f=np.zeros(40)
step=0
C=1


#One Step optimisation algorithm
while step<10000:
    (i,j)=Distinct_Two_Random(40)
    (beta_i_new,beta_j_new,f_new,b_new)=OneStepAlgo(i,j,X_train,y_train,f,beta,b,C)
    beta[i]=beta_i_new
    beta[j]=beta_j_new
    b=b_new
    f=f_new
    step=step+1

n=40
i=0
beta=beta.reshape((40,1))
alpa_SVM=np.dot(np.transpose(X_train),y_train*beta)
koja=np.dot(X_train,alpa_SVM)
b=-0.5*(max(koja[0:20])+min(koja[20:40]))



#Prediction SVM
n=1000
Confusion_matrix_SVM=np.zeros((2,2))
y_test_no=np.dot(X_test,alpa_SVM)+b
y_probabilities=1/(1+np.exp(-1*y_test_no))
y_predict=np.zeros(1000)
i=0
while i<n:
    if y_probabilities[i]==0.5:
        y_predict[i]=np.random.randint(2,size=1)
        if(y_predict[i]==0):
            y_predict[i]=-1
    elif y_probabilities[i]<0.5:
        y_predict[i]=-1
    else:
        y_predict[i]=1
    i=i+1
y0=y_predict[0:500]
y1=y_predict[500:1000]
Confusion_matrix_SVM[1,0]=np.sum((y0+1)/2)
Confusion_matrix_SVM[0,0]=500-Confusion_matrix_SVM[1,0]
Confusion_matrix_SVM[1,1]=np.sum((y1+1)/2)
Confusion_matrix_SVM[0,1]=500-Confusion_matrix_SVM[1,1]
acc_SVM=np.trace(Confusion_matrix_SVM)/n
print('acc on testing for SVM with C= ',C," is ",acc_SVM)

#Linear Regression(with gradient descent)

#Model Fitting with parameters
Convg_criterion=0.0000001
Max_Step=10000
Mu=0.00001

alpa0_linreg=np.random.normal(0,1,3)
alpa0_linreg=alpa0_linreg.reshape((3,1))
step=0
grad=2*(np.dot(np.dot(np.transpose(X_train_Reg),X_train_Reg),alpa0_linreg).reshape(3,1))-np.dot(np.transpose(X_train_Reg),y_train.reshape((40,1)))
alpa1_linreg=alpa0_linreg-Mu*grad

while np.linalg.norm(alpa1_linreg-alpa0_linreg)>Convg_criterion and step<Max_Step:
    alpa0_linreg=alpa1_linreg
    grad=2*(np.dot(np.dot(np.transpose(X_train_Reg),X_train_Reg),alpa0_linreg)-np.dot(np.transpose(X_train_Reg),y_train))
    alpa1_linreg=alpa0_linreg-Mu*grad
    step=step+1


#Validation
n=1000
Confusion_matrix_LinReg=np.zeros((2,2))
y_test_no=np.dot(X_test_Reg,alpa1_linreg)-0.5
y_probabilities=1/(1+np.exp(-1*y_test_no))
y_predict=np.zeros(1000)
i=0
while i<n:
    if y_probabilities[i]==0.5:
        y_predict[i]=np.random.randint(2,size=1)
        if(y_predict[i]==0):
            y_predict[i]=-1
    elif y_probabilities[i]<0.5:
        y_predict[i]=-1
    else:
        y_predict[i]=1
    i=i+1
y0=y_predict[0:500]
y1=y_predict[500:1000]
Confusion_matrix_LinReg[1,0]=np.sum((y0+1)/2)
Confusion_matrix_LinReg[0,0]=500-Confusion_matrix_LinReg[1,0]
Confusion_matrix_LinReg[1,1]=np.sum((y1+1)/2)
Confusion_matrix_LinReg[0,1]=500-Confusion_matrix_LinReg[1,1]
acc_LinReg=np.trace(Confusion_matrix_LinReg)/n
print("acc on testing for LinReg is ",acc_LinReg)



#Logistic Regression(with gradient descent)
Convg_criterion=0.000001
Max_Step=10000
Mu=10
alpa0_lr=np.random.normal(0,1,3)
alpa0_lr=alpa0_lr.reshape((3,1))
step=0
p=1/(1+np.exp(-1*np.dot(X_train_Reg,alpa0_lr)))
grad=np.dot(np.transpose(X_train_Reg),(p-y_train))
alpa1_lr=alpa0_lr-Mu*grad

while np.linalg.norm(alpa1_lr-alpa0_lr)>Convg_criterion and step<Max_Step:
    alpa0_lr=alpa1_lr
    p=1/(1+np.exp(-1*np.dot(X_train_Reg,alpa0_lr)))
    grad=np.dot(np.transpose(X_train_Reg),(p-y_train))
    alpa1_lr=alpa0_lr-Mu*grad
    step=step+1


#Validation
n=1000
Confusion_matrix_LR=np.zeros((2,2))
y_test_no=np.dot(X_test_Reg,alpa1_lr)-0.5
y_probabilities=1/(1+np.exp(-1*y_test_no))
y_predict=np.zeros(1000)
i=0
while i<n:
    if y_probabilities[i]==0.5:
        y_predict[i]=np.random.randint(2,size=1)
        if(y_predict[i]==0):
            y_predict[i]=-1
    elif y_probabilities[i]<0.5:
        y_predict[i]=-1
    else:
        y_predict[i]=1
    i=i+1
y0=y_predict[0:500]
y1=y_predict[500:1000]
Confusion_matrix_LR[1,0]=np.sum((y0+1)/2)
Confusion_matrix_LR[0,0]=500-Confusion_matrix_LR[1,0]
Confusion_matrix_LR[1,1]=np.sum((y1+1)/2)
Confusion_matrix_LR[0,1]=500-Confusion_matrix_LR[1,1]
acc_LR=np.trace(Confusion_matrix_LR)/n
print("acc on testing for Logistic reg is ",acc_LR)


#Plots on training data
pts_x0=np.linspace(-2,2,num=200)
pts_y_SVM=(-b+(-1)*alpa_SVM[0]*pts_x0)*(1/(alpa_SVM[1]))
pts_y_linreg=(0.5-alpa1_linreg[0]-(alpa1_linreg[1]*pts_x0))*(1/(alpa1_linreg[2]))
pts_y_lr=(0.5-alpa0_lr[0]-(alpa0_lr[1]*pts_x0))*(1/(alpa0_lr[2]))
plt.scatter(x_0[:,0],x_0[:,1],c='r')
plt.scatter(x_1[:,0],x_1[:,1],c='b')
plt.plot(pts_x0,pts_y_linreg,label='Lin Reg')
plt.plot(pts_x0,pts_y_SVM,label='SVM C=10 ')
plt.plot(pts_x0,pts_y_lr,label='Logistic Reg ')
plt.legend()
plt.show()






        
    
    
    