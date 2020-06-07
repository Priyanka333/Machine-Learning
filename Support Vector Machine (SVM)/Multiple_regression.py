#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[ ]:





# In[4]:


data = pd.read_csv("Startups_company.csv")

data.head()


# In[5]:


real_x =data.iloc[:,0:4].values
real_y = data.iloc[:,4].values
real_x


# In[6]:


le = LabelEncoder()
real_x[:,3] = le.fit_transform(real_x[:,3])
OneHE = OneHotEncoder()
real_x = OneHE.fit_transform(real_x).toarray()
real_x


# In[15]:


training_x, test_x, training_y, test_y = train_test_split(real_x, real_y,test_size=0.2, random_state=0)
test_x


# In[20]:


MLR = LinearRegression()
MLR.fit(training_x,training_y)


# In[23]:


pred_y = MLR.predict(test_x)
pred_y


# In[24]:


test_y


# In[ ]:




