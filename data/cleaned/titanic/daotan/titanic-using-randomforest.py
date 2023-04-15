#!/usr/bin/env python
# coding: utf-8

# # **Titanic using RandomForest**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')


# **Read dataset**

# In[ ]:


train_df = pd.read_csv("data/input/titanic/train.csv")
train_df.head()


# In[ ]:


train_df.describe()


# **Fill in missing values**

# In[ ]:


train_df.isnull().sum()


# In[ ]:


train_df["Age"].fillna(train_df.Age.mean(), inplace=True)
train_df["Embarked"].fillna("S", inplace=True)
train_df.isnull().sum()


# **Format only the data you need**

# In[ ]:


x_train = train_df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Parch', 'SibSp']]
x_train = pd.get_dummies(x_train)
x_train.head()


# In[ ]:


y_train = train_df[['Survived']]
y_train.head()


# **Learn in a RandomForest**

# In[ ]:


clf = RandomForestClassifier(random_state = 10, max_features='sqrt')
pipe = Pipeline([('classify', clf)])
param = {'classify__n_estimators':list(range(20, 30, 1)),
         'classify__max_depth':list(range(3, 10, 1))}
grid = GridSearchCV(estimator = pipe, param_grid = param, scoring = 'accuracy', cv = 10)