#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


import os


# In[5]:


os.getcwd()


# In[7]:


data=pd.read_csv("diabetes.csv")


# In[9]:


data.head()


# In[11]:


data.tail()


# In[13]:


data.info()


# In[15]:


data.describe()


# In[17]:


data.shape


# In[19]:


data.size


# In[21]:


data.ndim


# In[23]:


data.columns


# In[25]:


data.isna()


# In[27]:


data.isna().any()


# In[29]:


data.isna().sum()


# In[31]:


data['Age'].unique()


# In[33]:


data['Age'].duplicated()


# In[35]:


data['Age'].duplicated().sum()


# In[47]:


unique_values = data['Age'].unique()


# In[43]:


print(unique_values)


# In[49]:


unique_values = data['Age','Glucose'].unique()


# In[57]:


age_counts = data['Age'].duplicated().value_counts()


# In[61]:


print(age_counts)


# In[65]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[67]:


corr = data.corr()


# In[69]:


sns.heatmap(data.corr())


# In[71]:


plt.figure(figsize=(14,6))
sns.heatmap(data.corr())


# In[73]:


plt.figure(figsize=(14,6))
sns.heatmap(data.corr(),annot=True)


# In[77]:


sns.distplot(data,bins=20)
plt.show()


# In[ ]:




