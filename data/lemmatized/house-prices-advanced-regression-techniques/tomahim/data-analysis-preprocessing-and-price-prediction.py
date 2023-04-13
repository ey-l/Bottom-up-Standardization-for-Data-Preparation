import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style='white', color_codes=True)
pd.set_option('display.max_columns', 100)
import os
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', encoding='ISO-8859-1')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', encoding='ISO-8859-1')
_input1.head(n=20)
_input1.describe()

def filter_data(dataset):
    return dataset
_input1 = filter_data(_input1)
x_vars = ['PoolArea', 'GarageArea', 'YearBuilt', 'OverallQual', 'BedroomAbvGr', 'GrLivArea', 'TotRmsAbvGrd', 'TotalBsmtSF', 'MasVnrArea']
sns.pairplot(data=_input1, x_vars=x_vars, y_vars=['SalePrice'], size=3, kind='reg')
from math import floor
y_train = _input1['SalePrice']
del _input1['SalePrice']

def transform(dataset):
    dataset = pd.concat([dataset, pd.get_dummies(dataset['OverallQual'], prefix='Quality')], axis=1)
    dataset = pd.concat([dataset, pd.get_dummies(dataset['TotRmsAbvGrd'], prefix='NbRooms')], axis=1)
    dataset['TotalBsmtSF'] = dataset['TotalBsmtSF'].fillna(int(floor(dataset['TotalBsmtSF'].mean())), inplace=False)
    dataset['GarageArea'] = dataset['GarageArea'].fillna(int(floor(dataset['GarageArea'].mean())), inplace=False)
    dataset['MasVnrArea'] = dataset['MasVnrArea'].fillna(int(floor(dataset['MasVnrArea'].mean())), inplace=False)
    del dataset['OverallQual']
    del dataset['TotRmsAbvGrd']
    return dataset
X_train = filter_data(_input1)
X_train = transform(_input1)
X_test = transform(_input0)
predictor_cols = [col for col in X_train if col.startswith('Quality') or col.startswith('NbRooms') or col == 'GrLivArea' or (col == 'TotalBsmtSF') or (col == 'GarageArea')]
for col_name in predictor_cols:
    if col_name not in X_test.columns:
        X_test[col_name] = 0
X_train[predictor_cols].head()
from sklearn import linear_model
clf = linear_model.Lasso(alpha=0.1, max_iter=600000)