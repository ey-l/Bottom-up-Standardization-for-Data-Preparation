import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
warnings.filterwarnings('ignore')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
sample = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
print(train.shape)
print(test.shape)
print(sample.shape)
train['RoofStyle']
train.info()
train.drop(columns=['Id', 'SaleCondition', 'SaleType', 'MiscFeature', 'Fence', 'PoolQC', 'PavedDrive', 'GarageCond', 'GarageQual', 'GarageFinish', 'GarageType', 'FireplaceQu', 'Functional', 'KitchenQual', 'Electrical', 'CentralAir', 'HeatingQC', 'Heating', 'BsmtFinType2', 'BsmtFinType1', 'BsmtExposure', 'BsmtCond', 'BsmtQual', 'Foundation', 'ExterCond', 'ExterQual', 'MasVnrType', 'Exterior2nd', 'Exterior1st', 'RoofMatl', 'RoofStyle', 'HouseStyle', 'BldgType', 'Condition2', 'Condition1', 'Neighborhood', 'LandSlope', 'LotConfig', 'Utilities', 'LandContour', 'LotShape', 'Alley', 'Street', 'MSZoning'], inplace=True)
train.head()
train.info()
train.shape
test.drop(columns=['Id', 'SaleCondition', 'SaleType', 'MiscFeature', 'Fence', 'PoolQC', 'PavedDrive', 'GarageCond', 'GarageQual', 'GarageFinish', 'GarageType', 'FireplaceQu', 'Functional', 'KitchenQual', 'Electrical', 'CentralAir', 'HeatingQC', 'Heating', 'BsmtFinType2', 'BsmtFinType1', 'BsmtExposure', 'BsmtCond', 'BsmtQual', 'Foundation', 'ExterCond', 'ExterQual', 'MasVnrType', 'Exterior2nd', 'Exterior1st', 'RoofMatl', 'RoofStyle', 'HouseStyle', 'BldgType', 'Condition2', 'Condition1', 'Neighborhood', 'LandSlope', 'LotConfig', 'Utilities', 'LandContour', 'LotShape', 'Alley', 'Street', 'MSZoning'], inplace=True)
test.head()
test.info()
test.shape
train.info()
train['LotFrontage'].fillna(train['LotFrontage'].mean(), inplace=True)
train['MasVnrArea'].fillna(train['MasVnrArea'].mean(), inplace=True)
train['GarageYrBlt'].fillna(train['GarageYrBlt'].mean(), inplace=True)
train.info()
test['LotFrontage'].fillna(test['LotFrontage'].mean(), inplace=True)
test['MasVnrArea'].fillna(test['MasVnrArea'].mean(), inplace=True)
test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].mean(), inplace=True)
test['BsmtFinSF2'].fillna(test['BsmtFinSF2'].mean(), inplace=True)
test['BsmtUnfSF'].fillna(test['BsmtUnfSF'].mean(), inplace=True)
test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].mean(), inplace=True)
test['BsmtFullBath'].fillna(test['BsmtFullBath'].mean(), inplace=True)
test['BsmtHalfBath'].fillna(test['BsmtHalfBath'].mean(), inplace=True)
test['GarageYrBlt'].fillna(test['GarageYrBlt'].mean(), inplace=True)
test['GarageCars'].fillna(test['GarageCars'].mean(), inplace=True)
test['GarageArea'].fillna(test['GarageArea'].mean(), inplace=True)
test.info()
import matplotlib.pyplot as plt

plt.scatter(train['LotArea'], train['SalePrice'])
plt.scatter(train['MSSubClass'], train['SalePrice'])
plt.scatter(train['LotFrontage'], train['SalePrice'])
print(train.shape)
print(test.shape)
print(sample.shape)
X = train.drop(columns=['SalePrice'])
y = train[['SalePrice']]
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
X_trans = PolynomialFeatures(2).fit_transform(X)
test_trans = PolynomialFeatures(2).fit_transform(test)