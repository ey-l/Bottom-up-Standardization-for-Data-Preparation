import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
Id = _input0['Id']
_input1.head()
print(list(_input1.isna().sum()))
null_values_exceed = _input1.isna().sum(axis=0) > 600
cols_na_to_drop = []
for (index, bool_val) in zip(list(null_values_exceed.index), list(null_values_exceed)):
    if bool_val == True:
        cols_na_to_drop.append(index)
_input1 = _input1.drop(cols_na_to_drop, axis=1, inplace=False)
_input0 = _input0.drop(cols_na_to_drop, axis=1, inplace=False)
len(_input0.T)
len(_input1.T)
_input0.head()
_input1.shape
_input0.shape
na_cols = [i for i in _input1.columns if _input1[i].isna().sum() >= 1]
print(na_cols)
numeric_cols = _input1.select_dtypes(exclude='object').columns
len(numeric_cols)
categorical_cols = _input1.select_dtypes(include='object').columns
print(categorical_cols)
date_cols = [i for i in numeric_cols if i.__contains__('Yr') or i.__contains__('Year')]
print(date_cols)
numeric_cols = numeric_cols.drop(date_cols)
print(numeric_cols)
for i in date_cols:
    plt.plot(_input1[date_cols][i], _input1['SalePrice'], marker='.', linestyle='none')
    plt.xlabel(i)
    plt.ylabel('SalePrice')
_input1[numeric_cols].corr()['SalePrice'].sort_values(ascending=False)
corr_matrix = _input1[numeric_cols].corr().abs()
print('Columns Count: ', len(corr_matrix))
corr_matrix
cols_to_drop = []
threshold = 0.2
dt_corr = _input1[numeric_cols].corr().abs()['SalePrice'] < threshold
for (col_name, bool_Corr) in zip(list(dt_corr.index), list(dt_corr)):
    if bool_Corr == True:
        cols_to_drop.append(col_name)
print(cols_to_drop)
_input0.head()
_input1 = _input1.drop(cols_to_drop, inplace=False, axis=1)
_input0 = _input0.drop(cols_to_drop, inplace=False, axis=1)
_input1.head(25)
_input0.head(25)
numeric_cols = numeric_cols.drop(cols_to_drop)
_input1['LotFrontage'] = _input1['LotFrontage'].fillna(_input1['LotFrontage'].mean())
_input1[categorical_cols]
_input1[categorical_cols] = _input1[categorical_cols].fillna(_input1[categorical_cols].mode())
_input1[categorical_cols].isna().sum()
one_hot_cols = []
ordinal_cols = []
for col_name in _input1[categorical_cols].columns:
    if len(_input1[col_name].unique()) == 2:
        one_hot_cols.append(col_name)
    else:
        ordinal_cols.append(col_name)
print(one_hot_cols)
print(ordinal_cols)
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
_input1[ordinal_cols] = ordinal_encoder.fit_transform(_input1[ordinal_cols])
from sklearn.preprocessing import OneHotEncoder
one_encoder = OneHotEncoder(sparse=False)
for feature in one_hot_cols:
    _input1 = pd.concat([_input1, pd.get_dummies(_input1[feature], prefix=feature)], axis=1)
_input1.columns
_input1.shape
_input1 = _input1.drop(one_hot_cols, inplace=False, axis=1)
_input1.shape
_input0[ordinal_cols] = ordinal_encoder.fit_transform(_input0[ordinal_cols])
for feature in one_hot_cols:
    _input0 = pd.concat([_input0, pd.get_dummies(_input0[feature], prefix=feature)], axis=1)
_input0.columns
_input0.shape
_input0 = _input0.drop(one_hot_cols, inplace=False, axis=1)
_input0.shape
Y = _input1['SalePrice']
_input1 = _input1.drop('SalePrice', inplace=False, axis=1)
numeric_cols = numeric_cols.drop('SalePrice')
for feature in numeric_cols:
    _input1[feature] = _input1[feature] / _input1[feature].mean()
for feature in numeric_cols:
    _input0[feature] = _input0[feature] / _input0[feature].mean()
_input1.head(50)
_input0.head(50)
nans_cols_fi = [i for i in _input1.columns if _input1[i].isnull().any()]
nans_cols_fi
_input1[nans_cols_fi] = _input1[nans_cols_fi].apply(lambda x: x.fillna(x.value_counts().index[0]))
nans_cols_fi_numeric = [i for i in _input0[numeric_cols].columns if _input0[i].isnull().any()]
nans_cols_fi_numeric
_input0[nans_cols_fi_numeric] = _input0[nans_cols_fi_numeric].fillna(_input0[nans_cols_fi_numeric].mean())
nans_cols_fi_test_categorical = [i for i in _input0.columns if _input0[i].isnull().any()]
_input0[nans_cols_fi_test_categorical] = _input0[nans_cols_fi_test_categorical].apply(lambda x: x.fillna(x.value_counts().index[0]))
_input1.columns
_input0.columns
_input0['Utilities_NoSeWa'] = np.zeros(1459)
_input0.shape
_input0 = _input0.reindex(columns=_input1.columns)
from sklearn.linear_model import LinearRegression
model = LinearRegression()