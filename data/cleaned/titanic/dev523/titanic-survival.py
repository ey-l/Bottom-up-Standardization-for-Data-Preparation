#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing required libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as metrics
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


#reading training data
train = pd.read_csv("data/input/titanic/train.csv")


# In[ ]:


#first 5 rows
train.head()


# In[ ]:


#number of nulls in each column
train.isna().sum()


# In[ ]:


#checking data type 
train.dtypes


# ## Visualization

# In[ ]:


sns.histplot(data = train, x="Survived",color='r')


# In[ ]:


#Target variables quantitatives
sns.barplot(data = train, x="Survived", y="Age")


# In[ ]:


sns.boxplot(data = train, y="Age", x="Survived")


# In[ ]:


sns.barplot(data = train, x="Survived", y="Fare")


# In[ ]:


sns.boxplot(data = train, y="Fare", x="Survived")


# In[ ]:


#select columns to be used in training
colunas = ['Pclass','SibSp','Parch','Fare']
X = train[colunas]
y = train.Survived


# In[ ]:


#there is a need for pre-processing in the other fields
#split dataset into trainig and testing data from training data
train_X, val_X, train_y, val_y = train_test_split(X,y,random_state = 0)


# In[ ]:


model = DecisionTreeClassifier(random_state=1)