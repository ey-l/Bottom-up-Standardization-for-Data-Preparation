import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
_input0.head()

def missing_values(df):
    return df.isnull().values.sum()
missing_values(_input1)
missing_values(_input0)
_input1.info()
_input0.info()
_input1.describe().T
_input1.corr()['SalePrice'].sort_values()
_input1['OverallQual'].value_counts().sort_index()
sns.scatterplot(data=_input1, x='OverallQual', y='SalePrice')
plt.axhline(y=650000, c='r')
plt.axhline(y=200000, c='r')
_input1[(_input1['OverallQual'] > 8) & (_input1['SalePrice'] < 200000)][['OverallQual', 'SalePrice']]
_input1[(_input1['OverallQual'] > 8) & (_input1['SalePrice'] > 650000)][['OverallQual', 'SalePrice']]
sns.scatterplot(data=_input1, x='GrLivArea', y='SalePrice')
plt.axhline(y=650000, c='r')
plt.axhline(y=200000, c='r')
plt.axvline(x=4000, c='r')
_input1[(_input1['GrLivArea'] > 4000) & (_input1['SalePrice'] > 650000)][['GrLivArea', 'SalePrice']]
_input1[(_input1['GrLivArea'] > 4000) & (_input1['SalePrice'] < 200000)][['GrLivArea', 'SalePrice']]
sns.scatterplot(data=_input1, x='GarageCars', y='SalePrice')
plt.axhline(y=650000, c='r')
_input1[_input1['SalePrice'] > 650000][['GarageCars', 'SalePrice']]
drop0 = _input1[_input1['SalePrice'] > 650000].index
drop1 = _input1[(_input1['OverallQual'] > 8) & (_input1['SalePrice'] < 200000)].index
_input1 = _input1.drop(drop0, axis=0)
_input1 = _input1.drop(drop1, axis=0)
100 * (_input1.isnull().sum() / len(_input1)).sort_values()

def missing_value_percent(df):
    nan_percent = 100 * (df.isnull().sum() / len(df))
    nan_percent = nan_percent[nan_percent > 0].sort_values()
    return nan_percent

def fig_check(x):
    nan_percent = missing_value_percent(x)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=nan_percent.index, y=nan_percent)
    plt.xticks(rotation=90)
fig_check(_input1)
fig_check(_input0)
_input1 = _input1.dropna(axis=0, subset=['Electrical'])
cols = ['KitchenQual', 'GarageArea', 'GarageCars', 'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF2', 'SaleType', 'Exterior2nd', 'Exterior1st', 'BsmtFinSF1', 'Utilities', 'BsmtFullBath', 'BsmtHalfBath', 'Functional', 'MSZoning']
_input0 = _input0.dropna(axis=0, subset=cols)

def Mas(df):
    df['MasVnrType'] = df['MasVnrType'].fillna('None')
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
Mas(_input1)
Mas(_input0)
bsm = ['BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtExposure', 'BsmtFinType2']
for x in bsm:
    print(type(x))
_input1[bsm] = _input1[bsm].fillna('None')
_input0[bsm] = _input0[bsm].fillna('None')
fig_check(_input1)
garage = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
_input1[garage] = _input1[garage].fillna('None')
_input0[garage] = _input0[garage].fillna('None')
_input1['GarageYrBlt'] = _input1['GarageYrBlt'].fillna(_input1['GarageYrBlt'].mean())
_input0['GarageYrBlt'] = _input0['GarageYrBlt'].fillna(_input0['GarageYrBlt'].mean())
fig_check(_input1)
fig_check(_input0)
plt.figure(figsize=(10, 10))
sns.barplot(data=_input1, x='LotFrontage', y='Neighborhood', ci=None)
_input1.groupby('Neighborhood')['LotFrontage'].mean()
_input1['LotFrontage'] = _input1.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.mean()))
_input0['LotFrontage'] = _input0.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.mean()))
fig_check(_input1)
_input1['FireplaceQu'] = _input1['FireplaceQu'].fillna('None')
_input0['FireplaceQu'] = _input0['FireplaceQu'].fillna('None')
_input1 = _input1.drop(labels=['Id', 'Fence', 'Alley', 'MiscFeature', 'PoolQC'], axis=1)
_input0 = _input0.drop(labels=['Fence', 'Alley', 'MiscFeature', 'PoolQC'], axis=1)
nan_percent = missing_value_percent(_input1)
nan_percent
nan_percent = missing_value_percent(_input0)
nan_percent
_input1.head()
_input0.head()
_input1.info()
_input1['MSSubClass'] = _input1['MSSubClass'].astype(str)
_input0['MSSubClass'] = _input0['MSSubClass'].astype(str)
train_num = _input1.select_dtypes(exclude=object)
train_obj = _input1.select_dtypes(include=object)
train_num.info()
train_obj.info()
train_obj.shape
test_num = _input0.select_dtypes(exclude=object)
test_obj = _input0.select_dtypes(include=object)
test_num.info()
test_obj.info()
test_obj.shape
train_obj = pd.get_dummies(train_obj, drop_first=True)
test_obj = pd.get_dummies(test_obj, drop_first=True)
train_obj.shape
test_obj.shape
final0 = pd.concat([train_num, train_obj], axis=1)
final0.head()
final1 = pd.concat([test_num, test_obj], axis=1)
final1.head()
print('The Final shape of Training Set :', final0.shape)
print('The Final shape of Testing Set :', final1.shape)
X = final0.drop('SalePrice', axis=1)
y = final0['SalePrice']
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression
import xgboost as xgb
lin = LinearRegression()
XGB = xgb.XGBRegressor(use_label_encoder=False, eval_metric='rmse', n_estimators=5000, reg_alpha=0.1, reg_lambda=0.005, learning_rate=0.0125, max_depth=13, min_child_weight=4, gamma=0.04, subsample=0.7, colsample_bytree=0.6)