# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:55:24 2023

@author: Rik
"""

from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import pandas as pd
import os
from glob import iglob
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter

a = [] # contain all G values
vp= [] # contain all vp values

count=0
rootdir_glob=r'D:\Liquid nitrogen data of Au(bias and speed dependent)\hist1\***\**\*\G.*'
file_list = [f for f in iglob(rootdir_glob, recursive=True) if os.path.isfile(f)]
rootdir_glob_vp=r'D:\Liquid nitrogen data of Au(bias and speed dependent)\hist1\***\**\*\Vp.*'

file_list_vp = [f for f in iglob(rootdir_glob_vp, recursive=True) if os.path.isfile(f)]  

for f in file_list:
    print(count)
    
    df_G = pd.read_excel(f, sheet_name='Sheet1')
    df_vp = pd.read_excel(file_list_vp[count], sheet_name='Sheet1')
    count += 1
    for j in range(len(df_G.axes[1])):
    
            temp = []
            temp1 = []
            
            for i in range(len(df_G)):
                    
                    
                    if((df_G.iloc[i,j]) > 0.7 and (df_G.iloc[i,j]) < 1.4):
                            
                            temp.append(df_G.iloc[i,j])
                            temp1.append(df_vp.iloc[i,j])
            
            a.append(temp)  
            vp.append(temp1) 
            
a_corr=list(filter(None, a))
vp_corr= list(filter(None, vp))
a_50=[] # contain all trcaes with no of points > 100
vp_50=[] # corresponding vp values
index_list=[]
for index in range(len(a_corr)):
    if(len(a_corr[index])>100):
        a_50.append(a_corr[index])
        vp_50.append(vp_corr[index])
        index_list.append(index)
a_50_select=[] # select 100 points
vp_50_select=[]
for i in range(len(a_50)):       
    idx = np.round(np.linspace(0, len(a_50[i]) - 1, 100)).astype(int)     
    aa=np.array(a_50[i])
    vv=np.array(vp_50[i])
    a_50_select.append(list(aa[idx]))
    vp_50_select.append(list(vv[idx]))            
    
    
    
    
batch_size = 100
input_dim = len(a_50_select[0])
learning_rate = 1e-5
input_layer = Input(shape=(input_dim, ), name="input") #Input Layer
encoder = Dense (75, activation="sigmoid", activity_regularizer=regularizers.l1(learning_rate))(input_layer)#Encoder's first dense layer
encoder = Dense (50, activation="sigmoid",activity_regularizer=regularizers.l1(learning_rate))(encoder)#Encoder's second dense layer
encoder = Dense (25, activation="sigmoid", activity_regularizer=regularizers.l1(learning_rate))(encoder)# Code layer

decoder = Dense(50, activation="sigmoid", activity_regularizer=regularizers.l1(learning_rate))(encoder)# Decoder's first dense layer
decoder = Dense(75, activation="sigmoid", activity_regularizer=regularizers.l1(learning_rate))(decoder)# Decoder's second dense layer

decoder = Dense(input_dim, activation=None,activity_regularizer=regularizers.l1(learning_rate))(decoder)


autoencoder_1 = Model(inputs=input_layer, outputs=decoder)
autoencoder_1.compile(loss='MeanSquaredError',optimizer='adam')
satck_1 = autoencoder_1.fit(a_50_select, a_50_select,epochs=400,batch_size=batch_size)


from keras import backend as K
arr=np.array(a_50_select)
# with a Sequential model
get_3rd_layer_output = K.function([autoencoder_1.layers[0].input],[autoencoder_1.layers[3].output])
layer_output = get_3rd_layer_output([arr])[0]

# layer output from our run 
layer_output = pd.read_csv('layer_output.csv', header=None)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(layer_output)


kmeans = KMeans(
    init="k-means++",
    n_clusters=3,
    n_init=50,
    max_iter=500,
    random_state=42, tol=1e-6
)

kmeans.fit(scaled_features)


pca = PCA(n_components=2, random_state=42)
l = pca.fit(scaled_features)
output = l.transform(scaled_features)
centers = l.transform(kmeans.cluster_centers_)
labels=kmeans.labels_ # can be used to pull traces from different groups
plt.scatter(output[:, 0], output[:, 1], c=kmeans.labels_, cmap='tab20')

plt.scatter(centers[:, 0], centers[:, 1], s=50, color='yellow')


Counter(labels).keys() # equals to list(set(words))
Counter(labels).values()
