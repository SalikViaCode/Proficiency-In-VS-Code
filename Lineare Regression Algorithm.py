#!/usr/bin/env python
# coding: utf-8

# # Lineare Regression Algorithm

# In[5]:


#Aim : To perform and analysis of Linear Regression Algorithm


# In[7]:


#Name : Salik Sayyed
#Roll No : 44
#Class : 3B


# In[40]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[41]:


import os


# In[42]:


os.getcwd()


# In[43]:


os.chdir('C:\\Users\\salik\\DSS Practical')


# In[44]:


df=pd.read_csv("Salary.csv")


# In[52]:


df.head()


# In[54]:


df.tail()


# In[56]:


df.head(30)


# In[58]:


df.describe()


# In[60]:


df.shape


# In[62]:


df.size


# In[64]:


df.ndim


# In[66]:


df.isnull().sum()


# In[68]:


#Assiging values in X & Y
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


# In[70]:


print(X)


# In[72]:


print(y)


# In[74]:


#Splitting testdata into X_train,X_test,y_train,y_test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3,random_state=42)


# In[76]:


print(X_train)


# In[78]:


print(y_train)


# In[80]:


print (y_test)


# In[88]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)


# In[90]:


#Assigning Coefficient (slope) to m
m = lr.coef_ 


# In[92]:


print("Coefficient :" , m)


# In[94]:


#Assigning Y-intercept to a
c = lr.intercept_


# In[96]:


print("Intercept : ", c)


# In[98]:


lr.score(X_test,y_test) * 100


# In[ ]:




