#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns


# In[ ]:


df_Train=pd.read_csv('data/input/titanic/train.csv')
df_Test=pd.read_csv('data/input/titanic/test.csv')


# In[ ]:


df_Train.head()


# In[ ]:


df_Test.head()


# In[ ]:


df_Train.isnull().sum()


# In[ ]:


sns.heatmap(df_Train.isnull(),yticklabels=False,cbar=False)


# In[ ]:


df_Train.drop(['Cabin'],axis=1,inplace=True)
df_Train.drop(['Name'],axis=1,inplace=True)
df_Train.drop(['Ticket'],axis=1,inplace=True)
df_Train['Age']=df_Train['Age'].fillna(df_Train['Age'].mode()[0])
sns.heatmap(df_Train.isnull(),yticklabels=False,cbar=False)


# In[ ]:


sns.heatmap(df_Test.isnull(),yticklabels=False,cbar=False)


# In[ ]:


df_Test.drop(['Cabin'],axis=1,inplace=True)
df_Test.drop(['Name'],axis=1,inplace=True)
df_Test.drop(['Ticket'],axis=1,inplace=True)
df_Test['Age']=df_Test['Age'].fillna(df_Test['Age'].mode()[0])
df_Test['Fare']=df_Test['Fare'].fillna(df_Test['Fare'].mode()[0])
sns.heatmap(df_Test.isnull(),yticklabels=False,cbar=False)


# In[ ]:


df_Test.isnull().sum()


# In[ ]:


df_Train.describe()


# In[ ]:


df_Train.head()


# In[ ]:


S_Dummy=pd.get_dummies(df_Train['Sex'],drop_first=True)
Embarked_Dummy=pd.get_dummies(df_Train['Embarked'],drop_first=True)


# In[ ]:


df_Train=pd.concat([df_Train,S_Dummy,Embarked_Dummy],axis=1)
df_Train


# In[ ]:


df_Train.drop(['Sex','Embarked'],axis=1,inplace=True)
df_Train.head()


# In[ ]:


#Using Random forest for training data itself
from sklearn.model_selection import train_test_split
X=df_Train[['PassengerId','Pclass','Age','SibSp','Parch','Fare','male','Q','S']]
y=df_Train[['Survived']]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=5)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()