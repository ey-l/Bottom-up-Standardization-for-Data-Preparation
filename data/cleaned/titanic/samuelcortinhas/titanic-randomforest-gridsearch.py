#!/usr/bin/env python
# coding: utf-8

# # Background

# *The task is to predict which passangers survived the 1912 Titanic shipwreck given passanger information such as Age, Sex, Class etc.*

# # Libraries

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt



# # Data

# In[ ]:


# Read the data
X_train_full = pd.read_csv('data/input/titanic/train.csv', index_col='PassengerId')
X_test_full = pd.read_csv('data/input/titanic/test.csv', index_col='PassengerId')

# Number of rows and columns
print(X_train_full.shape)
print(X_test_full.shape)

# First 5 entries
X_train_full.head()


# **Check for missing values**

# In[ ]:


# Count null values
print(X_train_full.isnull().sum())
print('')
print(X_test_full.isnull().sum())


# **Labels and features**

# In[ ]:


# Labels
y = X_train_full.Survived

# Features
X_train_full.drop(['Survived'], axis=1, inplace=True)


# # Feature engineering

# **Title feature**

# In[ ]:


# Extract titles from 'Name' column
X_train_full['Title']=0
X_train_full['Title']=X_train_full.Name.str.extract('([A-Za-z]+)\.')

# Cross tabulation
pd.crosstab(X_train_full.Title,X_train_full.Sex).T


# In[ ]:


# Raplace rare titles by 'Rare'
X_train_full['Title'].replace(['Capt','Col','Countess','Don','Dr','Jonkheer','Lady','Major',
                           'Master','Miss','Mlle','Mme','Mr','Mrs','Ms','Rev','Sir'],
                           ['Rare','Rare','Rare','Rare','Rare','Rare','Rare',
                            'Rare','Master','Miss','Rare','Rare','Mr','Mrs','Rare',
                            'Rare','Rare'],inplace=True)

# Median age in each group
X_train_full.groupby('Title')['Age'].median()


# In[ ]:


# Countplot
sns.countplot(x="Title",
                   hue="Survived", 
                   data=pd.concat([X_train_full,y],axis=1),
                   palette = 'Blues_d')


# In[ ]:


# Assign missing age values to be median within each group
X_train_full.loc[(X_train_full.Age.isnull())&(X_train_full.Title=='Master'),'Age']=3.5
X_train_full.loc[(X_train_full.Age.isnull())&(X_train_full.Title=='Miss'),'Age']=21
X_train_full.loc[(X_train_full.Age.isnull())&(X_train_full.Title=='Mr'),'Age']=30
X_train_full.loc[(X_train_full.Age.isnull())&(X_train_full.Title=='Mrs'),'Age']=34.5
X_train_full.loc[(X_train_full.Age.isnull())&(X_train_full.Title=='Rare'),'Age']=44.5

# Check there are not missing values
X_train_full.Age.isnull().sum()


# **Repeat for test data**

# In[ ]:


# Repeat feature engineering for test data
X_test_full['Title']=0
X_test_full['Title']=X_test_full.Name.str.extract('([A-Za-z]+)\.') # extract titles

# Raplace rare titles by 'Rare'
X_test_full['Title'].replace(['Capt','Col','Countess','Don','Dr','Jonkheer','Lady','Major',
                           'Master','Miss','Mlle','Mme','Mr','Mrs','Ms','Rev','Sir','Dona'],
                           ['Rare','Rare','Rare','Rare','Rare','Rare','Rare',
                            'Rare','Master','Miss','Rare','Rare','Mr','Mrs','Rare',
                            'Rare','Rare','Rare'],inplace=True)

# Assign missing age values to be median within each group
X_test_full.loc[(X_test_full.Age.isnull())&(X_test_full.Title=='Master'),'Age']=3.5
X_test_full.loc[(X_test_full.Age.isnull())&(X_test_full.Title=='Miss'),'Age']=21
X_test_full.loc[(X_test_full.Age.isnull())&(X_test_full.Title=='Mr'),'Age']=30
X_test_full.loc[(X_test_full.Age.isnull())&(X_test_full.Title=='Mrs'),'Age']=34.5
X_test_full.loc[(X_test_full.Age.isnull())&(X_test_full.Title=='Rare'),'Age']=44.5

# Check there are not missing values
X_test_full.Age.isnull().sum()


# **HasCabin feature**

# In[ ]:


# Identify passangers with a recorded Cabin
X_train_full['HasCabin']=X_train_full['Cabin'].notnull()

# Repeat for test data
X_test_full['HasCabin']=X_test_full['Cabin'].notnull()


# In[ ]:


# Countplot
sns.countplot(x="HasCabin",
                   hue="Survived", 
                   data=pd.concat([X_train_full,y],axis=1),
                   palette = 'Blues_d')


# # Feature selection

# In[ ]:


# Select categorical columns to include in model
categorical_cols = ['Pclass', 'Sex', 'Embarked', 'HasCabin'] # Including 'Title' makes model worse

# Select numerical columns to include in model
numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch']

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()


# # Preprocessing data

# In[ ]:


# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='median')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Data preprocessing pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Transform the data
X_train = my_pipeline.fit_transform(X_train)


# # Grid Search

# In[ ]:


# Parameters grid
grid = {'n_estimators': [100, 125, 150, 175, 200, 225, 250], 
        'max_depth': [4, 6, 8, 10, 12]}

# Random Forest Classifier
clf=RandomForestClassifier(random_state=0)

# Grid Search with 4-fold cross validation
grid_model = GridSearchCV(clf,grid,cv=4)

# Train classifier with optimal parameters