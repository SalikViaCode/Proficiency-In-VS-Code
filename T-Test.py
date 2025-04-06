#!/usr/bin/env python
# coding: utf-8

# In[1]:


ages=[10,20,35,50,28,40,55,18,16,55,30,25,43,18,30,28,14,24,16,17,32,35,26,27,65,18,43,23,21,20,19,70]


# In[3]:


len(ages)


# In[5]:


type(ages)


# In[7]:


import numpy as np
ages_mean = np.mean(ages)
print(ages_mean)


# In[11]:


sample_size = 10
age_sample=np.random.choice(ages,sample_size)


# In[15]:


age_sample


# In[17]:


from scipy.stats import ttest_1samp


# In[27]:


ttest,p_value=ttest_1samp(age_sample,30)


# In[29]:


print(p_value)


# In[ ]:




