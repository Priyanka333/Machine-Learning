#!/usr/bin/env python
# coding: utf-8

# In[5]:



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[6]:


data = pd.read_csv("Company.csv")
data.head(10)


# In[13]:


real_x = data.iloc[:,0].values
real_y = data.iloc[:,1].values
real_x = real_x.reshape(-1,1)
real_y = real_y.reshape(-1,1)
print(real_x)


# In[14]:


training_x, testing_x,training_y, testing_y = train_test_split(real_x, real_y, test_size=0.3,random_state=0)


# In[15]:


Lin = LinearRegression()
Lin.fit(training_x, training_y)


# In[ ]:


Pred_y= Lin.predict(testing_x)


# In[16]:


#Y = b1x + b0
Lin.coef_


# In[17]:


Lin.intercept_


# In[18]:


9360.26128619*10.3+26777.3913412


# In[8]:


testing_y[3]


# In[9]:


Pred_y[3]


# In[16]:


plt.scatter(training_x,training_y, color='teal')
plt.plot(training_x,Lin.predict(training_x),color='red')
plt.title("Salary & Experience Training Plot")
plt.xlabel("Exp")
plt.ylabel("Salary")
plt.show()


# In[17]:


plt.scatter(testing_x,testing_y, color='teal')
plt.plot(training_x,Lin.predict(training_x),color='red')
plt.title("Salary & Experience Training Plot")
plt.xlabel("Exp")
plt.ylabel("Salary")
plt.show()


# In[ ]:




