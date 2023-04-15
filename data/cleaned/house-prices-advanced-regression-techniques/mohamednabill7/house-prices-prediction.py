import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train.head()
test.head()

def missing_values(df):
    return df.isnull().values.sum()
missing_values(train)
missing_values(test)
train.info()
test.info()
train.describe().T
train.corr()['SalePrice'].sort_values()
train['OverallQual'].value_counts().sort_index()
sns.scatterplot(data=train, x='OverallQual', y='SalePrice')
plt.axhline(y=650000, c='r')
plt.axhline(y=200000, c='r')
train[(train['OverallQual'] > 8) & (train['SalePrice'] < 200000)][['OverallQual', 'SalePrice']]
train[(train['OverallQual'] > 8) & (train['SalePrice'] > 650000)][['OverallQual', 'SalePrice']]
sns.scatterplot(data=train, x='GrLivArea', y='SalePrice')
plt.axhline(y=650000, c='r')
plt.axhline(y=200000, c='r')
plt.axvline(x=4000, c='r')
train[(train['GrLivArea'] > 4000) & (train['SalePrice'] > 650000)][['GrLivArea', 'SalePrice']]
train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 200000)][['GrLivArea', 'SalePrice']]
sns.scatterplot(data=train, x='GarageCars', y='SalePrice')
plt.axhline(y=650000, c='r')
train[train['SalePrice'] > 650000][['GarageCars', 'SalePrice']]
drop0 = train[train['SalePrice'] > 650000].index
drop1 = train[(train['OverallQual'] > 8) & (train['SalePrice'] < 200000)].index
train = train.drop(drop0, axis=0)
train = train.drop(drop1, axis=0)
100 * (train.isnull().sum() / len(train)).sort_values()

def missing_value_percent(df):
    nan_percent = 100 * (df.isnull().sum() / len(df))
    nan_percent = nan_percent[nan_percent > 0].sort_values()
    return nan_percent

def fig_check(x):
    nan_percent = missing_value_percent(x)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=nan_percent.index, y=nan_percent)
    plt.xticks(rotation=90)
fig_check(train)
fig_check(test)
train = train.dropna(axis=0, subset=['Electrical'])
cols = ['KitchenQual', 'GarageArea', 'GarageCars', 'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF2', 'SaleType', 'Exterior2nd', 'Exterior1st', 'BsmtFinSF1', 'Utilities', 'BsmtFullBath', 'BsmtHalfBath', 'Functional', 'MSZoning']
test = test.dropna(axis=0, subset=cols)

def Mas(df):
    df['MasVnrType'] = df['MasVnrType'].fillna('None')
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
Mas(train)
Mas(test)
bsm = ['BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtExposure', 'BsmtFinType2']
for x in bsm:
    print(type(x))
train[bsm] = train[bsm].fillna('None')
test[bsm] = test[bsm].fillna('None')
fig_check(train)
garage = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
train[garage] = train[garage].fillna('None')
test[garage] = test[garage].fillna('None')
train['GarageYrBlt'] = train['GarageYrBlt'].fillna(train['GarageYrBlt'].mean())
test['GarageYrBlt'] = test['GarageYrBlt'].fillna(test['GarageYrBlt'].mean())
fig_check(train)
fig_check(test)
plt.figure(figsize=(10, 10))
sns.barplot(data=train, x='LotFrontage', y='Neighborhood', ci=None)
train.groupby('Neighborhood')['LotFrontage'].mean()
train['LotFrontage'] = train.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.mean()))
test['LotFrontage'] = test.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.mean()))
fig_check(train)
train['FireplaceQu'] = train['FireplaceQu'].fillna('None')
test['FireplaceQu'] = test['FireplaceQu'].fillna('None')
train = train.drop(labels=['Id', 'Fence', 'Alley', 'MiscFeature', 'PoolQC'], axis=1)
test = test.drop(labels=['Fence', 'Alley', 'MiscFeature', 'PoolQC'], axis=1)
nan_percent = missing_value_percent(train)
nan_percent
nan_percent = missing_value_percent(test)
nan_percent
train.head()
test.head()
train.info()
train['MSSubClass'] = train['MSSubClass'].astype(str)
test['MSSubClass'] = test['MSSubClass'].astype(str)
train_num = train.select_dtypes(exclude=object)
train_obj = train.select_dtypes(include=object)
train_num.info()
train_obj.info()
train_obj.shape
test_num = test.select_dtypes(exclude=object)
test_obj = test.select_dtypes(include=object)
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