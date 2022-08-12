#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data_mnist=pd.read_csv(r'C:\Users\Aman Bhansali\Downloads\mnist_784_csv.csv')


# In[2]:


data_mnist.head()


# In[3]:


data_mnist.shape


# Splitted the data_mnist to two variables one containg the features and the other containing the labels

# In[4]:


features=data_mnist.drop('class', axis=1)
labels=data_mnist['class']


# Taking 5000 rows 

# In[5]:


features_=features[0:5000]


# In[6]:


features_


# In[7]:


features_list=features_.to_numpy()
features_list


# Getting the covariance matrix using .cov() 

# In[8]:


cov_matrix=features_.cov()
cov_matrix.shape


# Using the covariance matrix calculated the eigen vectors and eigen values

# In[58]:


eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)


# Arranged the eigen_vectors in descending order using the argsort first converted the eigen_values into descending order using idx and taking from reverse so that we may get descending ordered and then using same idx to take values from eigen_vectors

# In[25]:


idx = np.argsort(eigen_values)
idx=idx[::-1]
eigen_vectors = eigen_vectors[:, idx]
print(eigen_vectors)


# The first five eigen values

# In[11]:


print(eigen_values[0:5].real)


# In[28]:


data_=features_.to_numpy()
for i in range(5000):
    data_[i]=data_[i]-np.mean(data_[i])
data_.shape    


# 2.3 Reconstruct the image for different values of principal components. 
# 2.4 Visualize the reconstructed images made from the previous step and compare them with the original image. 
# 

# In[71]:


prin_comp=(10, 50, 100, 300, 700)
for i in prin_comp:
    print(i)
    eigen_vectors_=eigen_vectors[0:i].real
    eigen_vectors_.shape
    temp_feat = np.array(features_)
    df_= (eigen_vectors_ @ data_.transpose()).transpose()
    print(df_.shape)
    df_=df_.real
    temp = (df_ @ eigen_vectors_)
    print(temp.shape)
    X_= (df_ @ eigen_vectors_) + data_
    X_=X_.real
    X__=X_[4000].reshape(28, 28)
    Y__=X_[4006].reshape(28, 28)
    print(plt.imshow(X__, cmap='gray'))
    print(plt.show())
    print(plt.imshow(Y__, cmap='gray'))
    print(plt.show())
    temp


# In[14]:


data_=features.to_numpy()
data__4000=data_[4000].reshape(28, 28)
plt.imshow(data__4000, cmap='gray')
plt.show()


# In[15]:


data_=features.to_numpy()
data__4006=data_[4006].reshape(28, 28)
plt.imshow(data__4006, cmap='gray')
plt.show()


# 2.5 Visualize the residual images by subtracting the reconstructed image from the original image

# In[63]:


for i in prin_comp:
    print(i)
    eigen_vectors_=eigen_vectors[0:i].real
    eigen_vectors_.shape
    temp_feat = np.array(features_)
    df_= (eigen_vectors_ @ data_.transpose()).transpose()
    df_=df_.real
    temp = (df_ @ eigen_vectors_)
    X_= (df_ @ eigen_vectors_) + data_
    X_=X_.real
    X__=X_[4000].reshape(28, 28)
    Y__=X_[4006].reshape(28, 28)
    Z__4000=X__- data_[4000].reshape(28, 28)
    Z__4006=Y__- data_[4006].reshape(28, 28)
    print(plt.imshow(Z__4000, cmap='gray'))
    print(plt.show())
    print(plt.imshow(Z__4006, cmap='gray'))
    print(plt.show())
    


# Find the reconstruction error for each sample and plot them for a different number of principal components

# In[38]:


rms_error_4000=0
rms_error_4006=0
import math


# In[64]:


for i in prin_comp:
    print(i)
    eigen_vectors_=eigen_vectors[0:i].real
    eigen_vectors_.shape
    temp_feat = np.array(features_)
    df_= (eigen_vectors_ @ data_.transpose()).transpose()
    df_=df_.real
    temp = (df_ @ eigen_vectors_)
    X_= (df_ @ eigen_vectors_) + data_
    X_=X_.real
    X__=X_[4000].reshape(28, 28)
    Y__=X_[4006].reshape(28, 28)
    Z__4000=X__- data_[4000].reshape(28, 28)
    
    
    Z__4006=Y__- data_[4006].reshape(28, 28)
    flat_4000 = Z__4000.flatten()
    flat_4006 = Z__4006.flatten()
    for i in range(len(flat_4000)):
        
        rms_error_4000+=(flat_4000[i]**2)
    print(math.sqrt(rms_error_4000.sum())/784) 
    for i in range(len(flat_4006)):
        
        rms_error_4006+=(flat_4006[i]**2)
    print(math.sqrt(rms_error_4006.sum())/784) 


# In[69]:


rms_4000=[165.0570036298418, 166.2124496454606, 169.69140426771207, 177.5488612387833, 192.7107091075914]
comp=[10, 50, 100, 300, 700]
plt.plot(comp, rms_4000)
plt.show


# In[70]:


rms_4006=[2.364363418320797, 2.4209050146219315, 2.540191229827471, 2.804603411950273, 3.32254579283348]
comp_4006=[10, 50, 100, 300, 700]
plt.plot(comp_4006, rms_4006)
plt.show


# In[ ]:




