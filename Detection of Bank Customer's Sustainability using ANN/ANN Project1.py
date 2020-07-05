#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense


# In[4]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# In[5]:


dataset = pd.read_csv("Customer_Churn_Modelling.csv")
dataset.head()


# In[6]:


x = dataset.drop(labels=['CustomerId', 'Surname', 'RowNumber', 'Exited'], axis = 1)
y = dataset['Exited']


# In[7]:


x.head(10)


# In[8]:


y.head()


# In[9]:


from sklearn.preprocessing import LabelEncoder


# In[10]:


label1 = LabelEncoder()
x['Geography'] = label1.fit_transform(x['Geography'])


# In[11]:


x.head(10)


# In[12]:


label = LabelEncoder()
x['Gender'] = label1.fit_transform(x['Gender'])


# In[13]:


x.head(10)


# In[15]:


x = pd.get_dummies(x, drop_first=True, columns=['Geography'])
x.head()


# In[17]:


###Feature Standardization (In order to bring feature values in same scale)


# In[19]:


from sklearn.preprocessing import StandardScaler


# In[23]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0, stratify = y)


# In[24]:


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[25]:


x_train


# In[26]:


#Build ANN


# In[30]:


model = Sequential()
model.add(Dense(x.shape[1], activation='relu', input_dim = x.shape[1]))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation = 'sigmoid'))


# In[31]:


x.shape[1]


# In[32]:


model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])


# In[34]:


model.fit(x_train, y_train.to_numpy(), batch_size = 10, epochs = 10, verbose = 1)


# In[35]:


y_pred = model.predict_classes(x_test)


# In[36]:


y_pred


# In[37]:


y_test


# In[38]:


model.evaluate(x_test, y_test.to_numpy())


# In[39]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[40]:


confusion_matrix(y_test, y_pred)


# In[41]:


accuracy_score(y_test, y_pred)


# In[ ]:




