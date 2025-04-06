#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


import os


# In[5]:


os.getcwd()


# In[9]:


data=pd.read_csv("banglore.csv")


# In[11]:


data.head()


# In[13]:


data.tail()


# In[15]:


data.head(10)


# In[17]:


data.describe()


# In[21]:


data.shape


# In[45]:


data.info


# In[23]:


data.size


# In[25]:


data.ndim


# In[29]:


data.isnull().sum()


# In[35]:


X=data.drop("price",axis=1) #all atributes expect price


# In[37]:


y=data["price"]


# In[39]:


print(X)


# In[41]:


print(y)


# In[47]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3,random_state=50)


# In[53]:


print(X_train)


# In[55]:


print(y_train)


# In[57]:


print (y_test)


# In[3]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)


# In[ ]:




