import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
test_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
train_data.info()
test_data.info()
numerical = train_data.select_dtypes(include=['int64', 'float64'])
numerical_test = test_data.select_dtypes(include=['int64', 'float64'])
numerical
numerical = numerical.drop(['SalePrice', 'Id'], axis=1)
numerical_test = numerical_test.drop(['Id'], axis=1)
numerical
for i in range(33):
    print(numerical[numerical.columns[i]].value_counts().to_frame())
numerical_categorical = pd.DataFrame()
numerical_continuous = pd.DataFrame()
i = 0
while i < 36:
    x = numerical[numerical.columns[i]].value_counts().to_frame()
    if x.shape[0] <= 20:
        numerical_categorical = pd.concat([numerical_categorical, numerical[x.columns[0]]], axis=1)
        i = i + 1
    numerical_continuous = pd.concat([numerical_continuous, numerical[x.columns[0]]], axis=1)
    i = i + 1
numerical_categorical
numerical_continuous
numerical_categorical_test = pd.DataFrame()
numerical_continuous_test = pd.DataFrame()
i = 0
while i < numerical_categorical.shape[1]:
    numerical_categorical_test = pd.concat([numerical_categorical_test, numerical_test[numerical_categorical.columns[i]]], axis=1)
    i = i + 1
i = 0
while i < numerical_continuous.shape[1]:
    numerical_continuous_test = pd.concat([numerical_continuous_test, numerical_test[numerical_continuous.columns[i]]], axis=1)
    i = i + 1
numerical_categorical_test
numerical_continuous_test
categorical = train_data.select_dtypes(include='object')
categorical_test = test_data.select_dtypes(include='object')
categorical
categorical.info()
categorical = categorical.drop(['Alley', 'PoolQC', 'MiscFeature'], axis=1)
categorical_test = categorical_test.drop(['Alley', 'PoolQC', 'MiscFeature'], axis=1)
categorical
categorical_cols = pd.concat([numerical_categorical, categorical], axis=1)
categorical_cols_test = pd.concat([numerical_categorical_test, categorical_test], axis=1)
categorical_cols
encoding = OneHotEncoder()