#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Aim : To perform and Data analysis with Confusion matrix


# In[ ]:


#Name : Sayyed Salik Sagheer Jawwad
#Roll No : 44
#Class : 3B


# In[1]:


import pandas as pd 
import numpy as np


# # Data acquisitionuing Pandas 

# In[4]:


import os


# In[6]:


os.getcwd()


# In[8]:


data=pd.read_csv("heart - heart.csv")


# In[10]:


data.head()


# In[12]:


data.tail()


# In[14]:


data.info()


# In[16]:


data.describe()


# In[18]:


data.shape


# In[20]:


data.size


# In[22]:


data.ndim


# # Data preprocessing _ data cleaning _ missing value treatment

# In[25]:


# check Missing Value by record 

data.isna()


# In[27]:


data.isna().any()


# In[29]:


data.isna().sum()


# # Splitting of DataSet into train and Test

# In[32]:


x=data.drop("target", axis=1)
y=data["target"]


# In[34]:


#splitting the data into training and testing data sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2 ,random_state=42)


# In[36]:


x_train


# In[38]:


x_test


# In[40]:


y_train


# In[42]:


y_test


# # Logistic Regression

# In[45]:


data.head()


# In[47]:


from sklearn.linear_model import LogisticRegression


# In[49]:


log = LogisticRegression()
log.fit(x_train, y_train)


# In[51]:


y_pred1=log.predict(x_test)


# In[53]:


from sklearn.metrics import accuracy_score 


# In[55]:


accuracy_score (y_test,y_pred1)


# In[57]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# In[59]:


cm = confusion_matrix(y_test, y_pred1)

labels = np.unique(y_test)  # Get unique class labels
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

# Plot confusion matrix using seaborn
plt.figure(figsize=(6, 4))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', linewidths=1, linecolor='black')

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# In[ ]:




