import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
warnings.filterwarnings('ignore')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input2 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
print(_input1.shape)
print(_input0.shape)
print(_input2.shape)
_input1['RoofStyle']
_input1.info()
_input1 = _input1.drop(columns=['Id', 'SaleCondition', 'SaleType', 'MiscFeature', 'Fence', 'PoolQC', 'PavedDrive', 'GarageCond', 'GarageQual', 'GarageFinish', 'GarageType', 'FireplaceQu', 'Functional', 'KitchenQual', 'Electrical', 'CentralAir', 'HeatingQC', 'Heating', 'BsmtFinType2', 'BsmtFinType1', 'BsmtExposure', 'BsmtCond', 'BsmtQual', 'Foundation', 'ExterCond', 'ExterQual', 'MasVnrType', 'Exterior2nd', 'Exterior1st', 'RoofMatl', 'RoofStyle', 'HouseStyle', 'BldgType', 'Condition2', 'Condition1', 'Neighborhood', 'LandSlope', 'LotConfig', 'Utilities', 'LandContour', 'LotShape', 'Alley', 'Street', 'MSZoning'], inplace=False)
_input1.head()
_input1.info()
_input1.shape
_input0 = _input0.drop(columns=['Id', 'SaleCondition', 'SaleType', 'MiscFeature', 'Fence', 'PoolQC', 'PavedDrive', 'GarageCond', 'GarageQual', 'GarageFinish', 'GarageType', 'FireplaceQu', 'Functional', 'KitchenQual', 'Electrical', 'CentralAir', 'HeatingQC', 'Heating', 'BsmtFinType2', 'BsmtFinType1', 'BsmtExposure', 'BsmtCond', 'BsmtQual', 'Foundation', 'ExterCond', 'ExterQual', 'MasVnrType', 'Exterior2nd', 'Exterior1st', 'RoofMatl', 'RoofStyle', 'HouseStyle', 'BldgType', 'Condition2', 'Condition1', 'Neighborhood', 'LandSlope', 'LotConfig', 'Utilities', 'LandContour', 'LotShape', 'Alley', 'Street', 'MSZoning'], inplace=False)
_input0.head()
_input0.info()
_input0.shape
_input1.info()
_input1['LotFrontage'] = _input1['LotFrontage'].fillna(_input1['LotFrontage'].mean(), inplace=False)
_input1['MasVnrArea'] = _input1['MasVnrArea'].fillna(_input1['MasVnrArea'].mean(), inplace=False)
_input1['GarageYrBlt'] = _input1['GarageYrBlt'].fillna(_input1['GarageYrBlt'].mean(), inplace=False)
_input1.info()
_input0['LotFrontage'] = _input0['LotFrontage'].fillna(_input0['LotFrontage'].mean(), inplace=False)
_input0['MasVnrArea'] = _input0['MasVnrArea'].fillna(_input0['MasVnrArea'].mean(), inplace=False)
_input0['BsmtFinSF1'] = _input0['BsmtFinSF1'].fillna(_input0['BsmtFinSF1'].mean(), inplace=False)
_input0['BsmtFinSF2'] = _input0['BsmtFinSF2'].fillna(_input0['BsmtFinSF2'].mean(), inplace=False)
_input0['BsmtUnfSF'] = _input0['BsmtUnfSF'].fillna(_input0['BsmtUnfSF'].mean(), inplace=False)
_input0['TotalBsmtSF'] = _input0['TotalBsmtSF'].fillna(_input0['TotalBsmtSF'].mean(), inplace=False)
_input0['BsmtFullBath'] = _input0['BsmtFullBath'].fillna(_input0['BsmtFullBath'].mean(), inplace=False)
_input0['BsmtHalfBath'] = _input0['BsmtHalfBath'].fillna(_input0['BsmtHalfBath'].mean(), inplace=False)
_input0['GarageYrBlt'] = _input0['GarageYrBlt'].fillna(_input0['GarageYrBlt'].mean(), inplace=False)
_input0['GarageCars'] = _input0['GarageCars'].fillna(_input0['GarageCars'].mean(), inplace=False)
_input0['GarageArea'] = _input0['GarageArea'].fillna(_input0['GarageArea'].mean(), inplace=False)
_input0.info()
import matplotlib.pyplot as plt
plt.scatter(_input1['LotArea'], _input1['SalePrice'])
plt.scatter(_input1['MSSubClass'], _input1['SalePrice'])
plt.scatter(_input1['LotFrontage'], _input1['SalePrice'])
print(_input1.shape)
print(_input0.shape)
print(_input2.shape)
X = _input1.drop(columns=['SalePrice'])
y = _input1[['SalePrice']]
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
X_trans = PolynomialFeatures(2).fit_transform(X)
test_trans = PolynomialFeatures(2).fit_transform(_input0)