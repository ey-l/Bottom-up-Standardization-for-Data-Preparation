import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import preprocessing
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
print(_input1.shape)
_input1.head()
_input1.info()
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print(_input0.shape)
_input0.head()
_input0.info()
_input1.isnull().sum()
sb.heatmap(_input1.isnull(), yticklabels=False, cbar=False)
corr_matrix = _input1.corr()
corr_matrix['SalePrice'].sort_values(ascending=False)
_input1 = _input1.drop(['Id', 'PoolQC', 'Fence', 'FireplaceQu', 'MiscFeature', 'Alley'], axis=1, inplace=False)
_input0 = _input0.drop(['PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu', 'Alley'], axis=1, inplace=False)
print(_input1.shape)
_input0.shape
_input1.describe()
_input1['LotFrontage'] = _input1['LotFrontage'].fillna(_input1['LotFrontage'].mean())
_input1['BsmtCond'] = _input1['BsmtCond'].fillna(_input1['BsmtCond'].mode()[0])
_input1['BsmtQual'] = _input1['BsmtQual'].fillna(_input1['BsmtQual'].mode()[0])
_input1['GarageType'] = _input1['GarageType'].fillna(_input1['GarageType'].mode()[0])
_input1['GarageYrBlt'] = _input1['GarageYrBlt'].fillna(_input1['GarageYrBlt'].mode()[0])
_input1['GarageFinish'] = _input1['GarageFinish'].fillna(_input1['GarageFinish'].mode()[0])
_input1['GarageQual'] = _input1['GarageQual'].fillna(_input1['GarageQual'].mode()[0])
_input1['GarageCond'] = _input1['GarageCond'].fillna(_input1['GarageCond'].mode()[0])
_input1['MasVnrType'] = _input1['MasVnrType'].fillna(_input1['MasVnrType'].mode()[0])
_input1['MasVnrArea'] = _input1['MasVnrArea'].fillna(_input1['MasVnrArea'].mode()[0])
_input1['BsmtExposure'] = _input1['BsmtExposure'].fillna(_input1['BsmtExposure'].mode()[0])
_input1['BsmtFinType2'] = _input1['BsmtFinType2'].fillna(_input1['BsmtFinType2'].mode()[0])
_input1['BsmtFinType1'] = _input1['BsmtFinType1'].fillna(_input1['BsmtFinType1'].mode()[0])
_input0['LotFrontage'] = _input0['LotFrontage'].fillna(_input0['LotFrontage'].mean())
_input0['BsmtCond'] = _input0['BsmtCond'].fillna(_input0['BsmtCond'].mode()[0])
_input0['BsmtQual'] = _input0['BsmtQual'].fillna(_input0['BsmtQual'].mode()[0])
_input0['GarageType'] = _input0['GarageType'].fillna(_input0['GarageType'].mode()[0])
_input0['GarageYrBlt'] = _input0['GarageYrBlt'].fillna(_input0['GarageYrBlt'].mode()[0])
_input0['GarageFinish'] = _input0['GarageFinish'].fillna(_input0['GarageFinish'].mode()[0])
_input0['GarageQual'] = _input0['GarageQual'].fillna(_input0['GarageQual'].mode()[0])
_input0['GarageCond'] = _input0['GarageCond'].fillna(_input0['GarageCond'].mode()[0])
_input0['MasVnrType'] = _input0['MasVnrType'].fillna(_input0['MasVnrType'].mode()[0])
_input0['MasVnrArea'] = _input0['MasVnrArea'].fillna(_input0['MasVnrArea'].mode()[0])
_input0['BsmtExposure'] = _input0['BsmtExposure'].fillna(_input0['BsmtExposure'].mode()[0])
_input0['BsmtFinType2'] = _input0['BsmtFinType2'].fillna(_input0['BsmtFinType2'].mode()[0])
_input0['BsmtFinType1'] = _input0['BsmtFinType1'].fillna(_input0['BsmtFinType1'].mode()[0])
_input1.info()
_input1 = _input1.dropna(inplace=False)
_input1.shape
_input1['Age_House'] = _input1['YrSold'] - _input1['YearBuilt']
_input1['TotalBsmtBath'] = _input1['BsmtFullBath'] + _input1['BsmtHalfBath'] * 0.5
_input1['TotalBath'] = _input1['FullBath'] + _input1['HalfBath'] * 0.5
_input1['TotalSA'] = _input1['TotalBsmtSF'] + _input1['1stFlrSF'] + _input1['2ndFlrSF']
_input1.head()
corr_matrix = _input1.corr()
corr_matrix['SalePrice'].sort_values(ascending=False)
_input0['Age_House'] = _input0['YrSold'] - _input0['YearBuilt']
_input0['TotalBsmtBath'] = _input0['BsmtFullBath'] + _input0['BsmtHalfBath'] * 0.5
_input0['TotalBath'] = _input0['FullBath'] + _input0['HalfBath'] * 0.5
_input0['TotalSA'] = _input0['TotalBsmtSF'] + _input0['1stFlrSF'] + _input0['2ndFlrSF']
_input1.head()
_input1.hist(bins=50, figsize=(30, 35))
st_column = []
for i in _input1.columns:
    if _input1[i].dtypes == 'object':
        st_column.append(i)
print(st_column)
print(len(st_column))
plt.figure(figsize=(40, 38))
sb.heatmap(_input1.corr(), annot=True)
_input1 = _input1.drop(['1stFlrSF', 'GrLivArea', 'BsmtFullBath', 'FullBath', 'TotalBsmtSF'], axis=1, inplace=False)
_input0 = _input0.drop(['1stFlrSF', 'GrLivArea', 'BsmtFullBath', 'FullBath', 'TotalBsmtSF'], axis=1, inplace=False)
_input0 = _input0.fillna(0, inplace=False, axis=1)
_input0.shape
y = _input1['SalePrice']
_input1 = _input1.drop('SalePrice', axis=1, inplace=False)
X = pd.get_dummies(_input1[_input1.columns])
X.head()
from xgboost import XGBRegressor
_id = _input0['Id']
_input0 = _input0.drop('Id', axis=1, inplace=False)
x_test = pd.get_dummies(_input0[_input0.columns])
x_test.shape
unwanted = [col for col in X.columns if not col in x_test.columns]
print(unwanted)
X = X.drop(unwanted, axis=1, inplace=False)
X.shape
unwanted = [col for col in x_test.columns if not col in X.columns]
print(unwanted)
x_test = x_test.drop(unwanted, axis=1, inplace=False)
x_test.shape
xgbr = XGBRegressor(n_estimators=100)