#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from matplotlib import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Read the data
X_full = pd.read_csv('data/input/titanic/train.csv', index_col='PassengerId')
X_test_full = pd.read_csv('data/input/titanic/test.csv', index_col='PassengerId')

# separate target from predictor
y = X_full.Survived
X_full.drop(['Survived'], axis=1, inplace=True)

# Remove rows with missing target, separate target from predictors
# X_full.dropna(axis=0, subset=['Survived'], inplace=True)
# y = X_full.Survived
# X_full.drop(['Survived'], axis=1, inplace=True)

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, train_size=0.8, test_size=0.2, random_state=0)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in X_train_full.columns if
                    X_train_full[cname].nunique() < 10 and 
                    X_train_full[cname].dtype == "object"]


# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if 
                X_train_full[cname].dtype in ['int64', 'float64']]


# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()


# In[ ]:


# # Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# # Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))]) 

# # Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

model=LogisticRegression()
# model = XGBRegressor(n_estimators=1000, learning_rate=0.01)
# model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=1)
# model=XGBRegressor(booster='gbtree', colsample_bylevel=1,
#        colsample_bynode=1, colsample_bytree=1,
#        importance_type='gain', learning_rate=0.1, max_delta_step=0,
#        max_depth=3, n_estimators=100,
#        random_state=0,
#        reg_alpha=0, scale_pos_weight=1,
#        silent=None, verbosity=1)

 
 
# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

# Preprocessing of training data, fit model 