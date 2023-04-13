import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.describe()
_input1.info()
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams.update({'figure.max_open_warning': 0})
nc = []
cc = []
for col in _input1.columns:
    if _input1[col].dtype in ('int64', 'float64'):
        nc.append(_input1[col].name)
    else:
        cc.append(_input1[col].name)
ncd = _input1[nc]
ccd = _input1[cc]
"for i in nc:\n    sns.relplot(data=train, x=i,y='SalePrice')"
plt.figure(figsize=(60, 50))
ax = sns.heatmap(ncd.corr(), annot=True, fmt='.2f', cmap='cool')
print(ax)
for i in cc:
    ax = sns.catplot(x=i, data=_input1, kind='count', height=5, aspect=1.5)
    ax.set_xticklabels(rotation=30)
for i in cc:
    sns.catplot(x=i, y='SalePrice', data=_input1, kind='box')
obj = _input1.isnull().sum().sort_values(ascending=False)
for (key, value) in obj.iteritems():
    print(key, ',', value)
_input1 = _input1.drop(['PoolQC', 'MiscFeature', 'Fence', 'Alley'], axis=1)
for col in _input1:
    if (col in nc) & _input1[col].isnull().any():
        _input1[col] = _input1[col].fillna(_input1[col].mean(), inplace=False)
    if (col in cc) & _input1[col].isnull().any():
        _input1[col] = _input1[col].fillna(_input1[col].mode().iloc[0], inplace=False)
for col in cc:
    if ccd[col].value_counts().max() / ccd[col].count() > 0.95:
        _input1 = _input1.drop(col, axis=1, inplace=False)
from sklearn.preprocessing import OrdinalEncoder
ordinal = OrdinalEncoder()
for col in _input1.columns:
    if _input1[col].dtype not in ('int64', 'float64'):
        _input1[col] = ordinal.fit_transform(_input1[col].values.reshape(-1, 1))
for col in nc:
    if (_input1[col].corr(_input1['SalePrice']) < 0.1) & (_input1[col].corr(_input1['SalePrice']) > -0.1):
        _input1 = _input1.drop(col, axis=1, inplace=False)
Q1 = np.percentile(_input1['SalePrice'], 25, interpolation='midpoint')
Q3 = np.percentile(_input1['SalePrice'], 75, interpolation='midpoint')
IQR = Q3 - Q1
upper = np.where(_input1['SalePrice'] >= Q3 + 1.5 * IQR)
lower = np.where(_input1['SalePrice'] <= Q1 - 1.5 * IQR)
_input1 = _input1.drop(upper[0], errors='ignore', inplace=False)
_input1 = _input1.drop(lower[0], errors='ignore', inplace=False)
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
X = _input1.drop('SalePrice', axis=1)
Y = _input1['SalePrice']
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.3, random_state=0)
result = []
linearmodel = LinearRegression()