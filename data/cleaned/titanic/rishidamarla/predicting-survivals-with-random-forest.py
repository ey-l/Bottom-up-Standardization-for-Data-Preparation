#!/usr/bin/env python
# coding: utf-8

# # Random Forest Classification Predictions

# In this notebook I utilize a simple random forest classification algorithm to predict the passengers that survived the titatnic's crash and those who didn't.

# # Importing Libraries

# In[ ]:


# Importing all necessary Python Libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# # Data

# In[ ]:


# Uploading the training and testing datasets from Kaggle.
train_data = pd.read_csv("data/input/titanic/train.csv")
test_data = pd.read_csv("data/input/titanic/test.csv")


# In[ ]:


train_data.info()


# In[ ]:


# Checking for missing values.
train_data.isnull().sum()


# In[ ]:


# Removing the Cabin feature due to its extensive missing values.
train_data.drop('Cabin', axis=1, inplace=True)


# In[ ]:


# Replacing the missing embarked values with the value that appears the most frequently.
train_data["Embarked"].fillna(train_data['Embarked'].value_counts().idxmax(), inplace=True)


# In[ ]:


# Replacing the missing age values with the median age value.
train_data["Age"].fillna(train_data["Age"].median(skipna=True), inplace=True)


# In[ ]:


train_data.isnull().sum()


# In[ ]:


train_data.head()


# In[ ]:


# Encoding the sex values.
train_data['Sex'].replace("female", 0,inplace=True)
train_data['Sex'].replace("male", 1,inplace=True)


# In[ ]:


# Encoding the embarked values.
train_data['Embarked'].replace("S", 0,inplace=True)
train_data['Embarked'].replace("C", 1,inplace=True)
train_data['Embarked'].replace("Q", 2,inplace=True)


# In[ ]:


train_data.dtypes


# In[ ]:


test_data.isnull().sum()


# In[ ]:


test_data["Age"].fillna(test_data["Age"].median(skipna=True), inplace=True)
test_data["Fare"].fillna(test_data["Fare"].median(skipna=True), inplace=True)
test_data.drop('Cabin', axis=1, inplace=True)


# In[ ]:


test_data['Sex'].replace("female", 0,inplace=True)
test_data['Sex'].replace("male", 1,inplace=True)


# In[ ]:


test_data['Embarked'].replace("S", 0,inplace=True)
test_data['Embarked'].replace("C", 1,inplace=True)
test_data['Embarked'].replace("Q", 2,inplace=True)


# In[ ]:


test_data.dtypes


# In[ ]:


train_data.shape


# In[ ]:


test_data.shape


# In[ ]:


train_data.head()


# In[ ]:


outcome_data = train_data["Survived"]
train_data.drop(["Survived", "Ticket", "Name", "PassengerId"], axis=1, inplace=True)
test_data.drop(["Name","PassengerId","Ticket"], axis=1, inplace=True)


# # Random Forest Classification

# In[ ]:


from sklearn.model_selection import train_test_split

# Selecting the features and the outcome.
X = train_data.values
y = outcome_data.values

# Splitting the data into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report


# In[ ]:


# Initializing a RandomForestClassifier.
rf = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=4, max_features='auto',
                       max_leaf_nodes=5, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=15,
                       min_weight_fraction_leaf=0.0, n_estimators=350,
                       n_jobs=None, oob_score=False, random_state=1
                            , verbose=0,
                       warm_start=False)
