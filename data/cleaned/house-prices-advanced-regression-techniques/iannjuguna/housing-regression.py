import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train.describe()
train.info()
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams.update({'figure.max_open_warning': 0})

nc = []
cc = []
for col in train.columns:
    if train[col].dtype in ('int64', 'float64'):
        nc.append(train[col].name)
    else:
        cc.append(train[col].name)
ncd = train[nc]
ccd = train[cc]
"for i in nc:\n    sns.relplot(data=train, x=i,y='SalePrice')"
plt.figure(figsize=(60, 50))
ax = sns.heatmap(ncd.corr(), annot=True, fmt='.2f', cmap='cool')
print(ax)
for i in cc:
    ax = sns.catplot(x=i, data=train, kind='count', height=5, aspect=1.5)
    ax.set_xticklabels(rotation=30)
for i in cc:
    sns.catplot(x=i, y='SalePrice', data=train, kind='box')
obj = train.isnull().sum().sort_values(ascending=False)
for (key, value) in obj.iteritems():
    print(key, ',', value)
train = train.drop(['PoolQC', 'MiscFeature', 'Fence', 'Alley'], axis=1)
for col in train:
    if (col in nc) & train[col].isnull().any():
        train[col].fillna(train[col].mean(), inplace=True)
    if (col in cc) & train[col].isnull().any():
        train[col].fillna(train[col].mode().iloc[0], inplace=True)
for col in cc:
    if ccd[col].value_counts().max() / ccd[col].count() > 0.95:
        train.drop(col, axis=1, inplace=True)
from sklearn.preprocessing import OrdinalEncoder
ordinal = OrdinalEncoder()
for col in train.columns:
    if train[col].dtype not in ('int64', 'float64'):
        train[col] = ordinal.fit_transform(train[col].values.reshape(-1, 1))
for col in nc:
    if (train[col].corr(train['SalePrice']) < 0.1) & (train[col].corr(train['SalePrice']) > -0.1):
        train.drop(col, axis=1, inplace=True)
Q1 = np.percentile(train['SalePrice'], 25, interpolation='midpoint')
Q3 = np.percentile(train['SalePrice'], 75, interpolation='midpoint')
IQR = Q3 - Q1
upper = np.where(train['SalePrice'] >= Q3 + 1.5 * IQR)
lower = np.where(train['SalePrice'] <= Q1 - 1.5 * IQR)
train.drop(upper[0], errors='ignore', inplace=True)
train.drop(lower[0], errors='ignore', inplace=True)
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
X = train.drop('SalePrice', axis=1)
Y = train['SalePrice']
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.3, random_state=0)
result = []
linearmodel = LinearRegression()