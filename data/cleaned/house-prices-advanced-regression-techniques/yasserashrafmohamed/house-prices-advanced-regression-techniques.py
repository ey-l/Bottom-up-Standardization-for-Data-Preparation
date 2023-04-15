import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
train_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
Id = test_data['Id']
train_data.head()
print(list(train_data.isna().sum()))
null_values_exceed = train_data.isna().sum(axis=0) > 600
cols_na_to_drop = []
for (index, bool_val) in zip(list(null_values_exceed.index), list(null_values_exceed)):
    if bool_val == True:
        cols_na_to_drop.append(index)
train_data.drop(cols_na_to_drop, axis=1, inplace=True)
test_data.drop(cols_na_to_drop, axis=1, inplace=True)
len(test_data.T)
len(train_data.T)
test_data.head()
train_data.shape
test_data.shape
na_cols = [i for i in train_data.columns if train_data[i].isna().sum() >= 1]
print(na_cols)
numeric_cols = train_data.select_dtypes(exclude='object').columns
len(numeric_cols)
categorical_cols = train_data.select_dtypes(include='object').columns
print(categorical_cols)
date_cols = [i for i in numeric_cols if i.__contains__('Yr') or i.__contains__('Year')]
print(date_cols)
numeric_cols = numeric_cols.drop(date_cols)
print(numeric_cols)
for i in date_cols:
    plt.plot(train_data[date_cols][i], train_data['SalePrice'], marker='.', linestyle='none')
    plt.xlabel(i)
    plt.ylabel('SalePrice')

train_data[numeric_cols].corr()['SalePrice'].sort_values(ascending=False)
corr_matrix = train_data[numeric_cols].corr().abs()
print('Columns Count: ', len(corr_matrix))
corr_matrix
cols_to_drop = []
threshold = 0.2
dt_corr = train_data[numeric_cols].corr().abs()['SalePrice'] < threshold
for (col_name, bool_Corr) in zip(list(dt_corr.index), list(dt_corr)):
    if bool_Corr == True:
        cols_to_drop.append(col_name)
print(cols_to_drop)
test_data.head()
train_data.drop(cols_to_drop, inplace=True, axis=1)
test_data.drop(cols_to_drop, inplace=True, axis=1)
train_data.head(25)
test_data.head(25)
numeric_cols = numeric_cols.drop(cols_to_drop)
train_data['LotFrontage'] = train_data['LotFrontage'].fillna(train_data['LotFrontage'].mean())
train_data[categorical_cols]
train_data[categorical_cols] = train_data[categorical_cols].fillna(train_data[categorical_cols].mode())
train_data[categorical_cols].isna().sum()
one_hot_cols = []
ordinal_cols = []
for col_name in train_data[categorical_cols].columns:
    if len(train_data[col_name].unique()) == 2:
        one_hot_cols.append(col_name)
    else:
        ordinal_cols.append(col_name)
print(one_hot_cols)
print(ordinal_cols)
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
train_data[ordinal_cols] = ordinal_encoder.fit_transform(train_data[ordinal_cols])
from sklearn.preprocessing import OneHotEncoder
one_encoder = OneHotEncoder(sparse=False)
for feature in one_hot_cols:
    train_data = pd.concat([train_data, pd.get_dummies(train_data[feature], prefix=feature)], axis=1)
train_data.columns
train_data.shape
train_data.drop(one_hot_cols, inplace=True, axis=1)
train_data.shape
test_data[ordinal_cols] = ordinal_encoder.fit_transform(test_data[ordinal_cols])
for feature in one_hot_cols:
    test_data = pd.concat([test_data, pd.get_dummies(test_data[feature], prefix=feature)], axis=1)
test_data.columns
test_data.shape
test_data.drop(one_hot_cols, inplace=True, axis=1)
test_data.shape
Y = train_data['SalePrice']
train_data.drop('SalePrice', inplace=True, axis=1)
numeric_cols = numeric_cols.drop('SalePrice')
for feature in numeric_cols:
    train_data[feature] = train_data[feature] / train_data[feature].mean()
for feature in numeric_cols:
    test_data[feature] = test_data[feature] / test_data[feature].mean()
train_data.head(50)
test_data.head(50)
nans_cols_fi = [i for i in train_data.columns if train_data[i].isnull().any()]
nans_cols_fi
train_data[nans_cols_fi] = train_data[nans_cols_fi].apply(lambda x: x.fillna(x.value_counts().index[0]))
nans_cols_fi_numeric = [i for i in test_data[numeric_cols].columns if test_data[i].isnull().any()]
nans_cols_fi_numeric
test_data[nans_cols_fi_numeric] = test_data[nans_cols_fi_numeric].fillna(test_data[nans_cols_fi_numeric].mean())
nans_cols_fi_test_categorical = [i for i in test_data.columns if test_data[i].isnull().any()]
test_data[nans_cols_fi_test_categorical] = test_data[nans_cols_fi_test_categorical].apply(lambda x: x.fillna(x.value_counts().index[0]))
train_data.columns
test_data.columns
test_data['Utilities_NoSeWa'] = np.zeros(1459)
test_data.shape
test_data = test_data.reindex(columns=train_data.columns)
from sklearn.linear_model import LinearRegression
model = LinearRegression()