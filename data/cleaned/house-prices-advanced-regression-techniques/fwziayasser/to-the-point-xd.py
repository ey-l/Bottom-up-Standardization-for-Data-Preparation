import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import preprocessing
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
print(train_data.shape)
train_data.head()
train_data.info()
test_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print(test_data.shape)
test_data.head()
test_data.info()
train_data.isnull().sum()
sb.heatmap(train_data.isnull(), yticklabels=False, cbar=False)
corr_matrix = train_data.corr()
corr_matrix['SalePrice'].sort_values(ascending=False)
train_data.drop(['Id', 'PoolQC', 'Fence', 'FireplaceQu', 'MiscFeature', 'Alley'], axis=1, inplace=True)
test_data.drop(['PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu', 'Alley'], axis=1, inplace=True)
print(train_data.shape)
test_data.shape
train_data.describe()
train_data['LotFrontage'] = train_data['LotFrontage'].fillna(train_data['LotFrontage'].mean())
train_data['BsmtCond'] = train_data['BsmtCond'].fillna(train_data['BsmtCond'].mode()[0])
train_data['BsmtQual'] = train_data['BsmtQual'].fillna(train_data['BsmtQual'].mode()[0])
train_data['GarageType'] = train_data['GarageType'].fillna(train_data['GarageType'].mode()[0])
train_data['GarageYrBlt'] = train_data['GarageYrBlt'].fillna(train_data['GarageYrBlt'].mode()[0])
train_data['GarageFinish'] = train_data['GarageFinish'].fillna(train_data['GarageFinish'].mode()[0])
train_data['GarageQual'] = train_data['GarageQual'].fillna(train_data['GarageQual'].mode()[0])
train_data['GarageCond'] = train_data['GarageCond'].fillna(train_data['GarageCond'].mode()[0])
train_data['MasVnrType'] = train_data['MasVnrType'].fillna(train_data['MasVnrType'].mode()[0])
train_data['MasVnrArea'] = train_data['MasVnrArea'].fillna(train_data['MasVnrArea'].mode()[0])
train_data['BsmtExposure'] = train_data['BsmtExposure'].fillna(train_data['BsmtExposure'].mode()[0])
train_data['BsmtFinType2'] = train_data['BsmtFinType2'].fillna(train_data['BsmtFinType2'].mode()[0])
train_data['BsmtFinType1'] = train_data['BsmtFinType1'].fillna(train_data['BsmtFinType1'].mode()[0])
test_data['LotFrontage'] = test_data['LotFrontage'].fillna(test_data['LotFrontage'].mean())
test_data['BsmtCond'] = test_data['BsmtCond'].fillna(test_data['BsmtCond'].mode()[0])
test_data['BsmtQual'] = test_data['BsmtQual'].fillna(test_data['BsmtQual'].mode()[0])
test_data['GarageType'] = test_data['GarageType'].fillna(test_data['GarageType'].mode()[0])
test_data['GarageYrBlt'] = test_data['GarageYrBlt'].fillna(test_data['GarageYrBlt'].mode()[0])
test_data['GarageFinish'] = test_data['GarageFinish'].fillna(test_data['GarageFinish'].mode()[0])
test_data['GarageQual'] = test_data['GarageQual'].fillna(test_data['GarageQual'].mode()[0])
test_data['GarageCond'] = test_data['GarageCond'].fillna(test_data['GarageCond'].mode()[0])
test_data['MasVnrType'] = test_data['MasVnrType'].fillna(test_data['MasVnrType'].mode()[0])
test_data['MasVnrArea'] = test_data['MasVnrArea'].fillna(test_data['MasVnrArea'].mode()[0])
test_data['BsmtExposure'] = test_data['BsmtExposure'].fillna(test_data['BsmtExposure'].mode()[0])
test_data['BsmtFinType2'] = test_data['BsmtFinType2'].fillna(test_data['BsmtFinType2'].mode()[0])
test_data['BsmtFinType1'] = test_data['BsmtFinType1'].fillna(test_data['BsmtFinType1'].mode()[0])
train_data.info()
train_data.dropna(inplace=True)
train_data.shape
train_data['Age_House'] = train_data['YrSold'] - train_data['YearBuilt']
train_data['TotalBsmtBath'] = train_data['BsmtFullBath'] + train_data['BsmtHalfBath'] * 0.5
train_data['TotalBath'] = train_data['FullBath'] + train_data['HalfBath'] * 0.5
train_data['TotalSA'] = train_data['TotalBsmtSF'] + train_data['1stFlrSF'] + train_data['2ndFlrSF']
train_data.head()
corr_matrix = train_data.corr()
corr_matrix['SalePrice'].sort_values(ascending=False)
test_data['Age_House'] = test_data['YrSold'] - test_data['YearBuilt']
test_data['TotalBsmtBath'] = test_data['BsmtFullBath'] + test_data['BsmtHalfBath'] * 0.5
test_data['TotalBath'] = test_data['FullBath'] + test_data['HalfBath'] * 0.5
test_data['TotalSA'] = test_data['TotalBsmtSF'] + test_data['1stFlrSF'] + test_data['2ndFlrSF']
train_data.head()
train_data.hist(bins=50, figsize=(30, 35))

st_column = []
for i in train_data.columns:
    if train_data[i].dtypes == 'object':
        st_column.append(i)
print(st_column)
print(len(st_column))
plt.figure(figsize=(40, 38))
sb.heatmap(train_data.corr(), annot=True)
train_data.drop(['1stFlrSF', 'GrLivArea', 'BsmtFullBath', 'FullBath', 'TotalBsmtSF'], axis=1, inplace=True)
test_data.drop(['1stFlrSF', 'GrLivArea', 'BsmtFullBath', 'FullBath', 'TotalBsmtSF'], axis=1, inplace=True)
test_data.fillna(0, inplace=True, axis=1)
test_data.shape
y = train_data['SalePrice']
train_data.drop('SalePrice', axis=1, inplace=True)
X = pd.get_dummies(train_data[train_data.columns])
X.head()
from xgboost import XGBRegressor
_id = test_data['Id']
test_data.drop('Id', axis=1, inplace=True)
x_test = pd.get_dummies(test_data[test_data.columns])
x_test.shape
unwanted = [col for col in X.columns if not col in x_test.columns]
print(unwanted)
X.drop(unwanted, axis=1, inplace=True)
X.shape
unwanted = [col for col in x_test.columns if not col in X.columns]
print(unwanted)
x_test.drop(unwanted, axis=1, inplace=True)
x_test.shape
xgbr = XGBRegressor(n_estimators=100)