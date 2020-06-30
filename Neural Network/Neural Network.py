#!/usr/bin/env python
# coding: utf-8

# import tensorflow as tf

# In[2]:


import tensorflow as tf
from tensorflow import keras


# In[3]:


print(tf.__version__)


# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[5]:


mnist = keras.datasets.fashion_mnist


# In[6]:


type(mnist)


# In[7]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[8]:


x_train.shape, y_train.shape


# In[9]:


np.max(x_train)


# In[10]:


np.mean(x_train)


# In[11]:


y_train


# In[12]:


class_names = ['top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']


# In[13]:


#####Data Exploration


# In[14]:


x_train.shape


# In[15]:


x_test.shape


# In[16]:


plt.figure()
plt.imshow(x_train[0])
plt.colorbar()


# In[17]:


y_train


# In[18]:


x_train = x_train/255.0


# In[19]:


x_test = x_test/255.0


# In[20]:


plt.figure()
plt.imshow(x_train[1])
plt.colorbar()


# In[21]:


###Build the model with TF 2


# In[22]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense


# In[23]:


model = Sequential()
model.add(Flatten(input_shape = (28,28)))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))


# In[24]:


model.summary()


# In[25]:


####Model Compilation


# In[26]:


model.compile(optimizer='adam', loss= 'sparse_categorical_crossentropy', metrics = ['accuracy'])


# In[27]:


model.fit(x_train, y_train, epochs = 10)


# In[28]:


test_loss, test_acc = model.evaluate(x_test, y_test)
print(test_acc)


# In[29]:


from sklearn.metrics import accuracy_score


# In[30]:


y_pred = model.predict_classes(x_test)


# In[31]:


accuracy_score(y_test, y_pred)


# In[32]:


pred = model.predict(x_test)


# In[33]:


pred


# In[35]:


pred[0]


# In[36]:


np.argmax(pred[0])


# In[37]:


np.argmax(pred[1])


# In[38]:


y_pred


# In[ ]:





# In[ ]:




