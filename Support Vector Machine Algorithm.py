#!/usr/bin/env python
# coding: utf-8

# # Support Vector Machine Algorithm

# In[2]:


#Aim : To perform and analysis of SVM Algorithm


# In[4]:


#Name : Sayyed Salik Sagheer Jawwad
#Roll No : 44
#Class : 3B


# In[6]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# In[8]:


import os


# In[10]:


os.getcwd()


# In[12]:


os.chdir("C:\\Users\\salik\\DSS Practical")


# In[14]:


df=pd.read_csv("framingham.csv")


# In[16]:


df.head()


# In[18]:


df.tail()


# In[20]:


df.describe()


# In[22]:


df.info()


# In[24]:


df.isna().sum()


# In[26]:


df


# In[28]:


df['glucose'].fillna(value = df['glucose'].mean(),inplace=True)


# In[30]:


df['education'].fillna(value = df['education'].mean(),inplace=True)


# In[32]:


df['heartRate'].fillna(value = df['heartRate'].mean(),inplace=True)


# In[34]:


df['BMI'].fillna(value = df['BMI'].mean(),inplace=True)


# In[36]:


df['cigsPerDay'].fillna(value = df['cigsPerDay'].mean(),inplace=True)


# In[38]:


df['totChol'].fillna(value = df['totChol'].mean(),inplace=True)


# In[40]:


df['BPMeds'].fillna(value = df['BPMeds'].mean(),inplace=True)


# In[42]:


df.isna().sum()


# In[44]:


#Splitting the dependent and independent variables.
x = df.drop("TenYearCHD",axis=1)
y = df['TenYearCHD']


# In[46]:


x


# # Train Test Split
# 

# In[49]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# In[51]:


y_train


# # SVM Classifier

# In[54]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
svc=SVC()
svc.fit(x_test,y_test)
acc = svc.score(x_test,y_test)*100
print(acc)


# In[ ]:




