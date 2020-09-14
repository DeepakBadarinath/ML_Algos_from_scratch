# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 18:09:26 2020

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
import matplotlib.pyplot
from numpy import arange, meshgrid, sqrt

delta = 0.025
x, y = meshgrid(
    arange(-2, 2, delta),
    arange(-2, 2, delta)
)


def Kernel(ker,x,y):
    if(ker=='lin'):
        return np.dot(x,y)
    else:
        return np.dot(x,y)+(np.linalg.norm(x)*np.linalg.norm(y))*(np.linalg.norm(x)*np.linalg.norm(y))

def OneStepAlgo(ker,i,j,X,y,f,beta,b,C):
    delta=y[i]*((f[j]-y[j])-(f[i]-y[i]))
    s=y[i]*y[j]
    chi=Kernel(ker,X[i,:],X[i,:])+Kernel(ker,X[j,:],X[j,:])-2*Kernel(ker,X[i,:],X[j,:])
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
    f_new=f+(beta_j_new-beta[j])*y[j]*Kernel(ker,X[j,:],np.transpose(X))+(beta_i_new-beta[i])*y[i]*Kernel(ker,X[i,:],np.transpose(X))
    b_new=b-0.5*(f_new[i]-y[i]+f_new[j]-y[j])
    #print("Delta is ",delta," s is ",s," chi is ",chi," gamma is ",gamma," L is ",L," H is ",H," beta new is ",beta," f new is ",f_new," bnew is ",b_new)
    return (beta_i_new,beta_j_new,f_new,b_new)


def Distinct_Two_Random(n):
    i=np.random.randint(0,n)
    j=np.random.randint(0,n-1)
    if j>=i:
        j=j+1
    return (i,j)



#Training Data
R_0=np.sqrt(np.random.uniform(1,2,50))
R_1=np.sqrt(np.random.uniform(0,1,50))
theta_0=2*np.pi*np.random.uniform(0,1,50)
theta_1=2*np.pi*np.random.uniform(0,1,50)
x_0=np.c_[R_0*np.cos(theta_0),R_0*np.sin(theta_0)]
x_1=np.c_[R_1*np.cos(theta_1),R_1*np.sin(theta_1)]
X_train=np.r_[x_0,x_1]
y_train=np.r_[np.zeros(50)-1,np.ones(50)]
y_train=y_train.reshape(100,1)

#plot data
plt.scatter(x_0[:,0],x_0[:,1],c='r')
plt.scatter(x_1[:,0],x_1[:,1],c='b')

#Fit lin kernel
beta=np.zeros(100)
b=0
f=np.zeros(100)
step=0
C=10



while step<10000:
    (i,j)=Distinct_Two_Random(100)
    (beta_i_new,beta_j_new,f_new,b_new)=OneStepAlgo('lin',i,j,X_train,y_train,f,beta,b,C)
    beta[i]=beta_i_new
    beta[j]=beta_j_new
    b=b_new
    f=f_new
    step=step+1

n=100
i=0
beta=beta.reshape((100,1))
alpa_SVM_lin=np.dot(np.transpose(X_train),y_train*beta)
koja=np.dot(X_train,alpa_SVM_lin)
b_lin=-0.5*(max(koja[0:50])+min(koja[50:100]))



#Fit non linear Kernel
beta=np.zeros(100)
b=0
f=np.zeros(100)
step=0
C=10



while step<10000:
    (i,j)=Distinct_Two_Random(100)
    (beta_i_new,beta_j_new,f_new,b_new)=OneStepAlgo('quad',i,j,X_train,y_train,f,beta,b,C)
    beta[i]=beta_i_new
    beta[j]=beta_j_new
    b=b_new
    f=f_new
    step=step+1

n=100
i=0
beta=beta.reshape((100,1))

koja=np.dot(X_train,alpa_SVM_lin)
b_lin=-0.5*(max(koja[0:50])+min(koja[50:100]))



# Graphs
pts_x0=np.linspace(-2,2,num=300)
pts_y_lin=(-b_lin+(-1)*alpa_SVM_lin[0]*pts_x0)*(1/(alpa_SVM_lin[1]))
plt.plot(pts_x0,pts_y_lin,label='Separting line with C=10 ',c='g')
plt.legend()
















