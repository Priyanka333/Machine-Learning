#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv('spam.tsv', sep='\t')
df


# In[4]:


df.isnull().sum()


# In[5]:


len(df)


# In[6]:


df['label'].value_counts()


# In[7]:


###Balance Data


# In[8]:


ham = df[df['label']=='ham']
ham.head()


# In[9]:


spam = df[df['label']=='spam']
spam.head()


# In[10]:


ham.shape, spam.shape


# In[11]:


ham = ham.sample(spam.shape[0])


# In[12]:


ham.shape, spam.shape


# In[13]:


data = ham.append(spam, ignore_index=True)
data.tail()


# In[14]:


###Exploratory Data Analysis


# In[15]:


plt.hist(data[data['label']=='ham']['length'], bins = 100, alpha = 0.7)
plt.hist(data[data['label']=='spam']['length'], bins = 100, alpha = 0.7)
plt.show()


# In[16]:


plt.hist(data[data['label']=='ham']['punct'], bins = 100, alpha = 0.7)
plt.hist(data[data['label']=='spam']['punct'], bins = 100, alpha = 0.7)
plt.show()


# In[17]:


###Data Preparation


# In[18]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


# In[19]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[20]:


data.head()


# In[31]:


x_train, x_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size = 0.3, random_state= 0, shuffle= True, stratify=data['label'])


# In[22]:


y_train


# In[23]:


###Bag of Word Creation


# In[24]:


vectorizer = TfidfVectorizer()


# In[25]:


x_train = vectorizer.fit_transform(x_train)


# In[26]:


x_train.shape


# In[27]:


x_train


# In[28]:


###Pipeline and RF


# In[29]:


clf = Pipeline([('tfidf', TfidfVectorizer()),('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1))])


# In[32]:


clf.fit(x_train, y_train)


# In[34]:


y_pred = clf.predict(x_test)


# In[35]:


confusion_matrix(y_test, y_pred)


# In[36]:


print(classification_report(y_test, y_pred))


# In[37]:


accuracy_score(y_test, y_pred)


# In[38]:


clf.predict(["Hi, this is Priyanka"])


# In[39]:


clf.predict(["Congratulations, you have won a free tickets to USA"])


# In[ ]:




