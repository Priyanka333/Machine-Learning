#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


# In[17]:


data = pd.read_csv("ads.csv")
data.head(10)


# In[18]:


real_x = data.iloc[:,[2,3]].values
real_y = data.iloc[:,4].values


# In[14]:


training_x, test_x, training_y, test_y = train_test_split(real_x, real_y, test_size= 0.25, random_state= 0)


# In[20]:


s_c = StandardScaler()
training_x = s_c.fit_transform(training_x)
test_x = s_c.transform(test_x)


# In[22]:


cls = KNeighborsClassifier(n_neighbors= 5, metric='minkowski', p=2)
cls.fit(training_x, training_y)


# In[24]:


y_pred = cls.predict(test_x)
y_pred


# In[25]:


test_y


# In[27]:


c_m = confusion_matrix(test_y, y_pred)
c_m


# In[ ]:





# In[ ]:





# In[ ]:




