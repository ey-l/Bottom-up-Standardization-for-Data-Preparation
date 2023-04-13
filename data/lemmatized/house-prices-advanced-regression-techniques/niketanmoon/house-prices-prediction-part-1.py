import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1
_input1.info()
_input1.get('SalePrice').describe()
(f, ax) = plt.subplots(figsize=(16, 16))
sns.distplot(_input1.get('SalePrice'), kde=False)
corrmat = _input1.corr()
(f, ax) = plt.subplots(figsize=(16, 16))
sns.heatmap(corrmat, vmax=0.8, square=True)
plt.figure(figsize=(16, 16))
columns = corrmat.nlargest(10, 'SalePrice')['SalePrice'].index
correlation_matrix = np.corrcoef(_input1[columns].values.T)
sns.set(font_scale=1.25)
heat_map = sns.heatmap(correlation_matrix, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=columns.values, xticklabels=columns.values)
total = _input1.isna().sum().sort_values(ascending=False)
missing_data = pd.concat([total], axis=1, keys=['Total'])
missing_data.head(30)
_input1 = _input1.drop(missing_data[missing_data.get('Total') > 1].index, 1)
_input1 = _input1.drop(_input1.loc[_input1.get('Electrical').isna()].index)
_input1.isna().sum().max()
_input1.shape
categories = list(_input1.select_dtypes(['object']))
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categories)], remainder='passthrough')
X = _input1.drop(['Id', 'SalePrice'], axis=1)
X
print(X.shape)
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input0 = _input0.drop(missing_data[missing_data.get('Total') > 1].index, 1)
print(_input0.shape)
X = ct.fit_transform(X)
X.shape
X_test = _input0.drop(['Id'], axis=1)
_input0.info()
for i in X_test.isna().columns:
    if X_test.dtypes[i] != 'object':
        X_test[i] = X_test[i].fillna(X_test[i].mean())
    else:
        X_test[i] = X_test[i].fillna(X_test[i].mode()[0])
X_test.shape
X_test.isna().sum().max()
X_test = ct.transform(X_test)
X_test.shape
y = _input1.SalePrice
from sklearn.model_selection import train_test_split
(X_train, X_val, y_train, y_val) = train_test_split(X, y, train_size=0.8, random_state=1)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()