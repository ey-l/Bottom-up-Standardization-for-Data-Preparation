import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
train.head()
train['SalePrice'].describe()
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
test.head()
description = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/data_description.txt', sep='delimiter')
description.head(20)
train.isnull().sum().head(40)
train['LotFrontage'].fillna(train['LotFrontage'].median(), inplace=True)
train['GarageYrBlt'].fillna(value=0, inplace=True)
train['MasVnrArea'].fillna(train['MasVnrArea'].median(), inplace=True)
train['PoolQC'].fillna('Unknown', inplace=True)
train['Alley'].fillna('Unknown', inplace=True)
train['FireplaceQu'].fillna('Unknown', inplace=True)
train['MasVnrType'].fillna('Unknown', inplace=True)
train['Electrical'].fillna('Unknown', inplace=True)
train['BsmtFinType2'].fillna('Unknown', inplace=True)
train['BsmtFinType1'].fillna('Unknown', inplace=True)
train['BsmtExposure'].fillna('Unknown', inplace=True)
train['BsmtQual'].fillna('Unknown', inplace=True)
train['BsmtCond'].fillna('Unknown', inplace=True)
train['Fence'].fillna('Unknown', inplace=True)
train['MiscFeature'].fillna('Unknown', inplace=True)
train['GarageCond'].fillna('Unknown', inplace=True)
train['GarageQual'].fillna('Unknown', inplace=True)
train['GarageFinish'].fillna('Unknown', inplace=True)
train['GarageType'].fillna('Unknown', inplace=True)
train.isnull().sum()
test.isnull().sum().head(40)
test['LotFrontage'].fillna(test['LotFrontage'].median(), inplace=True)
test['GarageYrBlt'].fillna(value=0, inplace=True)
test['MSZoning'].fillna(value=0, inplace=True)
test['MasVnrArea'].fillna(test['MasVnrArea'].median(), inplace=True)
test['PoolQC'].fillna('Unknown', inplace=True)
test['Alley'].fillna('Unknown', inplace=True)
test['FireplaceQu'].fillna('Unknown', inplace=True)
test['MasVnrType'].fillna('Unknown', inplace=True)
test['Electrical'].fillna('Unknown', inplace=True)
test['BsmtFinType2'].fillna('Unknown', inplace=True)
test['BsmtFinType1'].fillna('Unknown', inplace=True)
test['BsmtExposure'].fillna('Unknown', inplace=True)
test['BsmtQual'].fillna('Unknown', inplace=True)
test['BsmtCond'].fillna('Unknown', inplace=True)
test['Fence'].fillna('Unknown', inplace=True)
test['MiscFeature'].fillna('Unknown', inplace=True)
test['GarageCond'].fillna('Unknown', inplace=True)
test['GarageQual'].fillna('Unknown', inplace=True)
test['GarageFinish'].fillna('Unknown', inplace=True)
test['GarageType'].fillna('Unknown', inplace=True)
test['SaleType'].fillna('Unknown', inplace=True)
test['Utilities'].fillna('Unknown', inplace=True)
test['Exterior1st'].fillna('Unknown', inplace=True)
test['Exterior2nd'].fillna('Unknown', inplace=True)
test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].median(), inplace=True)
test['BsmtFinSF1'].fillna(value=0, inplace=True)
test['BsmtUnfSF'].fillna(test['BsmtUnfSF'].median(), inplace=True)
test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].median(), inplace=True)
test['BsmtFullBath'].fillna(value=0, inplace=True)
test['BsmtHalfBath'].fillna(value=0, inplace=True)
test['KitchenQual'].fillna('Unknown', inplace=True)
test['Functional'].fillna('Unknown', inplace=True)
test['GarageCars'].fillna(test['GarageCars'].median(), inplace=True)
test['GarageArea'].fillna(test['GarageArea'].median(), inplace=True)
test.isnull().sum().tail(40)
import seaborn as sns
import matplotlib.pyplot as plt
sns.barplot(x='Street', y='SalePrice', data=train)
plt.xlabel('Street Type')
plt.ylabel('Sale Price')

sns.barplot(x='MoSold', y='SalePrice', data=train)
plt.xlabel('Month Sold')
plt.ylabel('Sale Price')

train[['MoSold', 'SalePrice']].corr()
sns.barplot(x='MSSubClass', y='SalePrice', data=train)

sns.barplot(x='MSZoning', y='SalePrice', data=train)

train[['LotFrontage', 'SalePrice']].corr()
train[['GarageYrBlt', 'SalePrice']].corr()
train[['MasVnrArea', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'FullBath', 'SalePrice']].corr()
sns.regplot(x='OverallQual', y='SalePrice', data=train)
plt.xlabel('Overall House Quality')
plt.ylabel('Selling Price')
plt.title('Linear Regression of House Quality')

sns.regplot(x='GrLivArea', y='SalePrice', data=train)
plt.xlabel('Ground Living Area')
plt.ylabel('Selling Price')
plt.title('Linear Regression of Living Area')

x_data = train[['OverallQual', 'GrLivArea', 'MasVnrArea', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF', 'FullBath']]
x_data.head()
x_data.shape
test_data = test[['OverallQual', 'GrLivArea', 'MasVnrArea', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF', 'FullBath']]
y_data = train['SalePrice']
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x_data, y_data, test_size=0.3, random_state=42)
lre = LinearRegression()