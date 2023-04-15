import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style='white', color_codes=True)
pd.set_option('display.max_columns', 100)
import os
train_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', encoding='ISO-8859-1')
test_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', encoding='ISO-8859-1')
train_data.head(n=20)
train_data.describe()

def filter_data(dataset):
    return dataset
train_data = filter_data(train_data)
x_vars = ['PoolArea', 'GarageArea', 'YearBuilt', 'OverallQual', 'BedroomAbvGr', 'GrLivArea', 'TotRmsAbvGrd', 'TotalBsmtSF', 'MasVnrArea']
sns.pairplot(data=train_data, x_vars=x_vars, y_vars=['SalePrice'], size=3, kind='reg')

from math import floor
y_train = train_data['SalePrice']
del train_data['SalePrice']

def transform(dataset):
    dataset = pd.concat([dataset, pd.get_dummies(dataset['OverallQual'], prefix='Quality')], axis=1)
    dataset = pd.concat([dataset, pd.get_dummies(dataset['TotRmsAbvGrd'], prefix='NbRooms')], axis=1)
    dataset['TotalBsmtSF'].fillna(int(floor(dataset['TotalBsmtSF'].mean())), inplace=True)
    dataset['GarageArea'].fillna(int(floor(dataset['GarageArea'].mean())), inplace=True)
    dataset['MasVnrArea'].fillna(int(floor(dataset['MasVnrArea'].mean())), inplace=True)
    del dataset['OverallQual']
    del dataset['TotRmsAbvGrd']
    return dataset
X_train = filter_data(train_data)
X_train = transform(train_data)
X_test = transform(test_data)
predictor_cols = [col for col in X_train if col.startswith('Quality') or col.startswith('NbRooms') or col == 'GrLivArea' or (col == 'TotalBsmtSF') or (col == 'GarageArea')]
for col_name in predictor_cols:
    if col_name not in X_test.columns:
        X_test[col_name] = 0
X_train[predictor_cols].head()
from sklearn import linear_model
clf = linear_model.Lasso(alpha=0.1, max_iter=600000)