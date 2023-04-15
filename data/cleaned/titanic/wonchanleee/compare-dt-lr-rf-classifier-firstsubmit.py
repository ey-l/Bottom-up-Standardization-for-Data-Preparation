#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


titanic_df = pd.read_csv('data/input/titanic/train.csv')


# ### **Train.csv preprocessing**

# In[ ]:


# I remove columns with duplicate semantics and category columns
titanic_df.drop(['Name', 'Cabin', 'Fare', 'Ticket', 'Embarked', 'PassengerId'], axis=1, inplace=True)
# I change category type str to int 
titanic_df.loc[titanic_df['Sex']=='male','Sex'] = 1
titanic_df.loc[titanic_df['Sex']=='female', 'Sex'] = 0
# 'Age'column's null data is replaced by mean of the column
titanic_df['Age'].fillna(titanic_df['Age'].mean(), inplace=True)


# In[ ]:


# I split the data into feature columns and target columns.  
train_features = titanic_df.drop('Survived', axis=1)
train_target = titanic_df['Survived']


# ### **Find Proper Classifier**

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

DT = DecisionTreeClassifier()
LR = LogisticRegression()
RF = RandomForestClassifier(n_estimators=1000)

model = [DT, LR, RF]

# DecisionTree
parameters_dt = {
    'max_depth':[1, 3, 5, 10],
    'min_samples_leaf':[1, 3, 5, 10]
}
grid_cv_dt = GridSearchCV(model[0], param_grid=parameters_dt, scoring='accuracy', cv=5)