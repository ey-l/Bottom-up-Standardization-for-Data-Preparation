#!/usr/bin/env python
# coding: utf-8

# ## Problem: Titanic - Machine Learning from Disaster

# ### This Notebook Covers:
# * Imports
# * Data Extraction
# * Data Preprocessing
# * Handle missing values of Age using Iterative Imputer
# * Convert Features as per it's appropriate dtype
# * Split Train Dataset into Features & Target
# * Train Test Split - For Validation
# * Feature Selection
# * Model Validation and Hyper Parameter Tunning
# * Model Training & Prediction
# * Prediction Distribution
# 

# ## Imports

# In[ ]:


import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, classification_report, precision_score, f1_score, roc_auc_score, accuracy_score


# ## Data Extraction

# In[ ]:


train_data = pd.read_csv("data/input/titanic/train.csv")
train_data.drop(['Name', 'Ticket', 'PassengerId', 'Cabin'], axis=1, inplace=True)
print(train_data.shape)
train_data.head()


# In[ ]:


test_data = pd.read_csv("data/input/titanic/test.csv")
test_data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
print(test_data.shape)
test_data.head()


# ## Data PreProcessing

# In[ ]:


# Sex Mapping 
sex_mapping = {'male':0, 'female':1}
train_data.Sex = train_data.Sex.map(sex_mapping)
test_data.Sex = test_data.Sex.map(sex_mapping)
 
# Map Embarked with numerical values - use in filling missing values of age using iterative imputer 
embarked_mapping = {'C':0, 'Q':1, 'S':2}
train_data.Embarked = train_data.Embarked.map(embarked_mapping)
test_data.Embarked = test_data.Embarked.map(embarked_mapping)

# Fill missing values of embarked and fare with median value
train_data['Embarked'].fillna(value=train_data['Embarked'].median(), inplace=True)
test_data['Embarked'].fillna(value=train_data['Embarked'].median(), inplace=True)
train_data['Fare'].fillna(value=train_data['Fare'].median(), inplace=True)
test_data['Fare'].fillna(value=train_data['Fare'].median(), inplace=True)

# Create New Feature Family Size
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1 # 1 for childer with nanny
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1

# Converts Embarked into dummy
train_data = pd.get_dummies(train_data)
test_data = pd.get_dummies(test_data)


# In[ ]:


train_data.isnull().sum()


# In[ ]:


test_data.isnull().sum()


# ## Handle missing values of Age using Iterative Imputer

# In[ ]:


useless_features = ['Survived', 'PassengerId']
useful_features = [i for i in train_data.columns if i not in useless_features]

# Fill missing value for Age using Iterative Imputer
imputer = IterativeImputer(max_iter=25, random_state=42)

train_data_imptr = imputer.fit_transform(train_data[useful_features])
train_data_imtr = pd.DataFrame(train_data_imptr, columns = useful_features)
train_data = train_data.drop(useful_features, axis=1)
train_data = pd.concat([train_data, train_data_imtr], axis=1)

test_data_imptr = imputer.transform(test_data[useful_features])
test_data_imtr = pd.DataFrame(test_data_imptr, columns= useful_features)
test_data = test_data.drop(useful_features, axis=1)
test_data = pd.concat([test_data, test_data_imtr], axis=1)


# In[ ]:


train_data.isnull().sum()


# In[ ]:


test_data.isnull().sum()


# ## Convert Features as per it's appropriate dtype

# In[ ]:


train_data['Survived'] = train_data['Survived'].astype(int)
train_data['Pclass'] = train_data['Pclass'].astype(int)
train_data['SibSp'] = train_data['SibSp'].astype(int)
train_data['Parch'] = train_data['Parch'].astype(int)
train_data['Embarked'] = train_data['Embarked'].astype(int)
train_data['FamilySize'] = train_data['FamilySize'].astype(int)

test_data['PassengerId'] = test_data['PassengerId'].astype(int)
test_data['Pclass'] = test_data['Pclass'].astype(int)
test_data['SibSp'] = test_data['SibSp'].astype(int)
test_data['Parch'] = test_data['Parch'].astype(int)
test_data['Embarked'] = test_data['Embarked'].astype(int)
test_data['FamilySize'] = test_data['FamilySize'].astype(int)

train_data.shape, test_data.shape


# In[ ]:


# Distribution of Target Variable
train_data['Survived'].hist()


# In[ ]:


train_data.head()


# ## Split Train Dataset into Features & Target

# In[ ]:


y = train_data['Survived']
X = train_data.drop(['Survived'], axis=1)
X_test = test_data.drop(['PassengerId'], axis=1)
X.shape, y.shape, X_test.shape


# ## Train Test Split - For Validation

# In[ ]:


X_train, X_valid, Y_train, Y_valid = train_test_split(X, y, random_state=42, test_size=0.25)
X_train.shape, X_valid.shape, Y_train.shape, Y_valid.shape


# ## Feature Selection

# In[ ]:


from sklearn.feature_selection import SelectFromModel
sel = SelectFromModel(RandomForestClassifier())