#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data=pd.read_csv("D:\\Machine Learning DataSet\\Diabetes\\diabetes_csv.csv")


# In[3]:


data.head(10)


# In[4]:


data.describe()


# In[5]:


data.isnull().sum()


# In[6]:


diabets_map={"tested_positive":1, "tested_negative":0}


# In[7]:


data['class']=data['class'].map(diabets_map)


# In[8]:


data.head(5)


# In[9]:


data.columns


# In[10]:


data[['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age']]=data[['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age']].replace(0,np.NaN)


# In[11]:


data.isnull().sum()


# In[12]:


data.fillna(data.mean(), inplace=True)


# In[13]:


data.isnull().sum()


# In[14]:


data.shape


# In[15]:


data.hist(grid=False, bins=25, figsize=(50,50))


# In[16]:


X=data.drop(['class'],axis=1)
y=data['class']


# In[17]:


X.shape


# In[18]:


y.shape


# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


x_train, x_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=0)


# In[21]:


x_train.shape


# In[22]:


y_train.shape


# In[23]:


from sklearn.preprocessing import MinMaxScaler


# In[24]:


sc=MinMaxScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)


# In[25]:


from sklearn.ensemble import RandomForestClassifier


# In[27]:


random_forest=RandomForestClassifier(random_state=20)
random_forest.fit(x_train, y_train)


# In[28]:


y_pred=random_forest.predict(x_test)


# In[29]:


y_pred


# In[30]:


from sklearn import metrics


# In[35]:


print("Accuracy ={0:.3f}".format(metrics.accuracy_score(y_test,y_pred)))


# In[ ]:




