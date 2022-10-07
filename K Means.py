# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 12:40:41 2022

@author: DELL


"""
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn.cluster import KMeans

df=pd.read_csv("E://TMC//K MEANS.csv")
df.info()
df.describe()
df1=df[["Item Desp","DeliveryType","Payable Amount"]]

def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)
df1_norm=norm_func(df1.iloc[:,2:])
TWSS=[]
k=list(range(2,9))

for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df1_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
plt.plot(k,TWSS,'ro-')
plt.xlabel("No_of_Clusters")
plt.ylabel("Total_within_SS")

model=KMeans(n_clusters=6)
model.fit(df1_norm)
model.labels_
mb=pd.Series(model.labels_)
df["Cluster"]=mb

df.head()
df=df.iloc[:,[7,1,0,2,3,4,5,6]]
df.iloc[:,2:8].groupby(df.Cluster).mean()
df.to_csv("KMeans-TMC.csv",encoding='utf-8')
import os
os.getcwd()



