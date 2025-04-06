#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Aim : To perform and analysis of Naive Bayes Algorithm


# In[3]:


#Name : Sayyed Salik Sagheer Jawwad
#Roll No : 44
#Class : 3B


# In[5]:


import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# In[7]:


os.getcwd()


# In[15]:


df=pd.read_csv('heart - heart.csv')


# In[17]:


df.head()


# In[19]:


df.tail()


# In[21]:


df.info()


# In[23]:


df.size


# In[25]:


df.shape


# In[27]:


df.isna().sum()


# In[29]:


df.describe()


# In[33]:


df.ndim


# In[35]:


data_dup = df.duplicated().any()


# In[37]:


data_dup


# In[39]:


data=df.drop_duplicates()


# In[41]:


data_dup =df.duplicated().any()


# In[43]:


data_dup


# In[45]:


x=df.drop("target", axis=1)
y=df["target"]


# In[47]:


#splitting the data into training and testing data sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2 ,random_state=42)


# In[49]:


x_train


# In[51]:


x_test


# In[53]:


y_train


# In[55]:


y_test


# In[57]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score 


# In[59]:


nb_classifier = GaussianNB()
nb_classifier.fit(x_train, y_train)


# In[61]:


y_pred = nb_classifier.predict(x_test)


# In[63]:


accuracy_score (y_test,y_pred)


# In[65]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

labels = np.unique(y_test)  # Get unique class labels
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

# Plot confusion matrix using seaborn
plt.figure(figsize=(6, 4))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', linewidths=1, linecolor='black')

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# In[67]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score


# In[69]:


# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Precision
precision = precision_score(y_test, y_pred, average='weighted')
print(f'Precision: {precision:.4f}')

# Recall
recall = recall_score(y_test, y_pred, average='weighted')
print(f'Recall: {recall:.4f}')

# Error Rate
error_rate = 1 - accuracy
print(f'Error Rate: {error_rate:.4f}')

# Classification report
print("Classification Report:")
print(classification_report(y_test,y_pred))


# In[71]:


from sklearn.model_selection import KFold, cross_val_score


# In[73]:


# Define K-Fold Cross Validation
k = 5  # Number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Perform Cross Validation
scores = cross_val_score(nb_classifier, x, y, cv=kf, scoring='accuracy')

# Print results
print(f'Cross-validation scores: {scores}')
print(f'Mean accuracy: {scores.mean():.4f}')


# In[ ]:




