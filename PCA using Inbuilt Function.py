#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data_mnist=pd.read_csv(r'C:\Users\Aman Bhansali\Downloads\mnist_784_csv.csv')


# After downloading the mnist dataset we observe that the dataset contains the information of pixels of the digits and the class contains the corresponding value of digit 

# In[2]:


data_mnist.head()


# 1.2 This is the description of the dataset 

# In[3]:


data_mnist.describe()


# In[4]:


data_mnist.shape


# 1.3 Visualize any one of the images in the dataset by reshaping the data (28 x 28).
# 

# It had 785 columns of which one was the class so I dropped the class part and stored rest in the variable features and then stored the class under labels  

# In[5]:


features=data_mnist.drop('class', axis=1)
labels=data_mnist['class']


# Now since our data was a dataframe so to reshape it we needed to convert it to list/array for which I used numpy and we got a 2D list and each list inside this contained 784 values and I took a random list like the 4th index and converted its data into 28X28 and then using imshow visualized the image. The total rows are 70,000 with columns=785

# In[6]:


data_=features.to_numpy()
data__=data_[4].reshape(28, 28)
print(data_)


# In[7]:


plt.imshow(data__)


# Also looked what was the label at index 4

# In[8]:


labels[4]


# 1.4 
# i. Performing dimensionality reduction, passing in the number of principal components an finding the variance of each component
# 

# Since a variance of greater than 95% is said to be good so I used the number of components = 157 and got a total variance of 0.9512709918101425 that is greater than 0.95

# In[9]:


from sklearn.decomposition import PCA
pr_ca = PCA(n_components=157)
prin_comp = pr_ca.fit_transform(features)


# In[10]:


var=pr_ca.explained_variance_ratio_
print(var)


# In[11]:


print(sum(var))


# ii. I gave the value of 0.94 and on passing I got the number of components that should be used equal to 134

# In[12]:


from sklearn.decomposition import PCA
pca = PCA(.94)
principal_Comp = pca.fit_transform(features)


# In[13]:


pca.n_components_


# Now on taking components = 134 we get a total variance by calculating the sum of the variances of all components= 0.9402630989652367 i.e greater than .94
# 

# In[14]:


var=pca.explained_variance_ratio_
print(var)
print(sum(var))


# In[ ]:




