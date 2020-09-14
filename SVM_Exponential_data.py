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
    #print("Delta is ",delta," s is ",s," chi is ",chi," gamma is ",gamma," L is ",L," H is ",H," beta new is ",beta," f new is ",f_new," bnew is ",b_new)
    return (beta_i_new,beta_j_new,f_new,b_new)


def Distinct_Two_Random(n):
    i=np.random.randint(0,n)
    j=np.random.randint(0,n-1)
    if j>=i:
        j=j+1
    return (i,j)


x_0=np.random.exponential(scale=0.25,size=(20,2))
x_1=np.random.exponential(scale=2,size=(20,2))
x_train=np.r_[x_0,x_1]
X_train=x_train
plt.scatter(x_0[:,0],x_0[:,1],c='r')
plt.scatter(x_1[:,0],x_1[:,1],c='b')

zo=np.zeros(20)-1
oz=np.ones(20)
oz=oz.reshape((20,1))
zo=zo.reshape((20,1))
y_train=np.r_[zo,oz]

beta=np.zeros(40)
b=0
f=np.zeros(40)
step=0
C=10



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
pts_x0=np.linspace(0,6,num=200)
pts_y0=(-b+(-1)*alpa_SVM[0]*pts_x0)*(1/(alpa_SVM[1]))
plt.plot(pts_x0,pts_y0,label='Separting line with C=10 ',c='g')
plt.legend()
plt.show()






        
    
    
    