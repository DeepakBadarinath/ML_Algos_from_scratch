# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 10:18:21 2020

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


"""z=np.random.randint(3,size=10000)
i=0
counter=0
while i<9998:
    if z[i]==2 and z[i+1]==0 and z[i+2]==1:
        counter=counter+1
    i=i+1
print(counter)"""
a = np.random.multivariate_normal([1.5,0],[[1,0],[0,1]],10)
b = np.random.multivariate_normal([1.5,0],[[1,0],[0,1]],10)
plt.scatter(a[:,0],a[:,1],c='r')
plt.scatter(b[:,0],b[:,1],c='b')


s=np.random.randint(10,size=100)
x2=a[s]+np.random.multivariate_normal([0,0],[[0.25,0],[0,0.25]],100)
s2=np.random.randint(10,size=100)
yi=a[s]+np.random.multivariate_normal([0,0],[[0.25,0],[0,0.25]],100)
x=np.r_[x2,yi]
plt.scatter(x2[:,0],x2[:,1],c='r')
plt.scatter(yi[:,0],yi[:,1],c='b')
X=np.c_[np.ones(200),x]

zo=np.zeros(100)
oz=np.ones(100)
oz=oz.reshape((100,1))
zo=zo.reshape((100,1))
y=np.r_[zo,oz]
X=np.array(X)
y=np.array(y)


alpa=np.linalg.solve(np.dot(np.transpose(X),X),np.dot(np.transpose(X),y))
pts_x=np.linspace(-3,4,num=200)
pts_y=(0.5-alpa[0]-(alpa[1]*pts_x))*(1/(alpa[2]))
plt.scatter(x2[:,0],x2[:,1],c='r')
plt.scatter(yi[:,0],yi[:,1],c='b')
plt.plot(pts_x,pts_y,c='g')


n=200
X_test=X
C=np.zeros((2,2))
y_test_no=np.dot(X_test,alpa)-0.5
y_test=np.sign(y_test_no)
i=0
while i<n:
    if y_test[i]==0:
        y_test[i]=np.random.randint(2,size=1)
        #print("Heloo")
    if y_test[i]==-1:
        y_test[i]=0
    i=i+1
y0=y_test[0:100]
y1=y_test[100:200]
C[1,0]=np.sum(y0)
C[0,0]=100-C[1,0]
C[1,1]=np.sum(y1)
C[0,1]=100-C[1,1]
acc=np.trace(C)/n
print('acc on training is ', acc)

#Testing on Test set
a = np.random.multivariate_normal([1.5,0],[[1,0],[0,1]],10)
b = np.random.multivariate_normal([1.5,0],[[1,0],[0,1]],10)
s=np.random.randint(10,size=500)
x2_test=a[s]+np.random.multivariate_normal([0,0],[[0.25,0],[0,0.25]],500)
s2=np.random.randint(10,size=500)
yi_test=a[s]+np.random.multivariate_normal([0,0],[[0.25,0],[0,0.25]],500)
x_test=np.r_[x2_test,yi_test]
X_test=np.c_[np.ones(1000),x_test]
zo=np.zeros(500)
oz=np.ones(500)
oz=oz.reshape((500,1))
zo=zo.reshape((500,1))
y_actual=np.r_[zo,oz]
X_test=np.array(X_test)
y_actual=np.array(y_actual)


pts_x=np.linspace(-3,4,num=200)
pts_y=(0.5-alpa[0]-(alpa[1]*pts_x))*(1/(alpa[2]))
plt.scatter(x2_test[:,0],x2_test[:,1],c='r')
plt.scatter(yi_test[:,0],yi_test[:,1],c='b')
plt.plot(pts_x,pts_y,c='g')


n=1000
C=np.zeros((2,2))
y_test_no=np.dot(X_test,alpa)-0.5
y_test=np.sign(y_test_no)
i=0
while i<n:
    if y_test[i]==0:
        y_test[i]=np.random.randint(2,size=1)
        #print("Heloo")
    if y_test[i]==-1:
        y_test[i]=0
    i=i+1
y0=y_test[0:100]
y1=y_test[100:200]
C[1,0]=np.sum(y0)
C[0,0]=500-C[1,0]
C[1,1]=np.sum(y1)
C[0,1]=500-C[1,1]
acc=np.trace(C)/n
print('acc on testing is ', acc)


