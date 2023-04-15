#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv('data/input/titanic/train.csv')
df = df.set_index('PassengerId' , drop = True)
df.head()


# In[3]:


dft = pd.read_csv('data/input/titanic/test.csv')
dft = dft.set_index('PassengerId' , drop = True)
dft.head()


# In[4]:


df.Ticket = pd.to_numeric(df.Ticket.str.split().str[-1] , errors='coerce')
df.info()


# In[5]:


dft.Ticket = pd.to_numeric(dft.Ticket.str.split().str[-1] , errors='coerce')
dft.info()


# In[6]:


df = df.drop(['Name' ,  'Cabin'] , axis = 1)


# In[7]:


dft = dft.drop(['Name' ,  'Cabin'] , axis = 1)


# In[8]:


df.head()


# In[9]:


dft.head()


# In[10]:


df = pd.get_dummies(df , columns= ['Sex' , 'Embarked'])


# In[11]:


dft = pd.get_dummies(dft , columns= ['Sex' , 'Embarked'])


# In[12]:


df.head()


# In[13]:


dft.head()


# In[14]:


df['Age'] = df['Age'].fillna((df['Age'].median()))
df['Ticket'] = df['Ticket'].fillna((df['Ticket'].median()))
df.info()


# In[15]:


dft['Age'] = dft['Age'].fillna((dft['Age'].median()))
dft['Fare'] = dft['Fare'].fillna((dft['Fare'].median()))
dft.info()


# In[16]:


y = df['Survived']
df = df.drop(['Survived' , 'Sex_male' ,'Embarked_S'] , axis = 1)
X = df


# In[17]:


dft = dft.drop(['Sex_male', 'Embarked_S'] , axis = 1)


# In[18]:



XTEST = dft
XTEST.head()


# In[19]:


X.head()


# In[20]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# In[21]:


from sklearn.model_selection import cross_val_score


# In[22]:


from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier( )
scores = cross_val_score(clf, X_scaled, y, cv=5).mean()
scores


# In[23]:


from xgboost import XGBClassifier
clf = XGBClassifier(colsample_bylevel= 0.9,
                    colsample_bytree = 0.8, 
                    gamma=0.99,
                    max_depth= 5,
                    min_child_weight= 1,
                    n_estimators= 10,
                    nthread= 4,
                    random_state= 2,
                    silent= True)
scores = cross_val_score(clf, X_scaled, y, cv=5).mean()
scores


# In[24]:

