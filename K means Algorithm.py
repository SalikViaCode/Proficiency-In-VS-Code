#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Aim: To perform and find the accuracy of K means algorithm


# In[3]:


#Name : Sayyed Salik Sagheer Jawwad
#Roll No : 44
#Class : 3B


# In[5]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for statistical data visualization
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.datasets import make_blobs
import warnings
warnings.filterwarnings('ignore')
import os


# In[7]:


os.getcwd()


# In[9]:


df=pd.read_csv("framingham.csv")


# In[11]:


df.head()


# In[13]:


df.info()


# In[15]:


df.size


# In[17]:


df.shape


# In[19]:


df.describe()


# In[21]:


from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


# In[23]:


X, y = make_blobs(n_samples=1000, centers=3, n_features=2, random_state=42)


# In[25]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X)


# In[27]:


kmeans.cluster_centers_


# In[ ]:




