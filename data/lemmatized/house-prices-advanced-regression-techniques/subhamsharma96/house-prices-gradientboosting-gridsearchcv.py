import pandas as pd
import numpy as np
import os
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
import warnings

def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train_parent = _input1
test_parent = _input0
_input1 = _input1.drop('Id', axis=1)
_input0 = _input0.drop('Id', axis=1)
threshold = 0.4 * len(_input1)
df = pd.DataFrame(len(_input1) - _input1.count(), columns=['count'])
df.index[df['count'] > threshold]
_input1 = _input1.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)
_input0 = _input0.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)
_input1['SalePrice'].describe()
_input1.select_dtypes(include=np.number).columns
for col in ('MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold'):
    _input1[col] = _input1[col].fillna(0)
    _input0[col] = _input0[col].fillna('0')
_input1.select_dtypes(exclude=np.number).columns
for col in ('MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition'):
    _input1[col] = _input1[col].fillna('None')
    _input0[col] = _input0[col].fillna('None')
_input1[_input1.isnull().any(axis=1)]
_input0[_input0.isnull().any(axis=1)]
train = _input1
test = _input0
train['train'] = 1
test['train'] = 0
combined = pd.concat([train, test])
ohe_data_frame = pd.get_dummies(combined, columns=['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition'])
train_df = ohe_data_frame[ohe_data_frame['train'] == 1]
test_df = ohe_data_frame[ohe_data_frame['train'] == 0]
train_df = train_df.drop(['train'], axis=1, inplace=False)
test_df = test_df.drop(['train', 'SalePrice'], axis=1, inplace=False)
_input1 = train_df
_input0 = test_df
X_train = _input1.drop('SalePrice', axis=1)
Y_train = _input1['SalePrice']
X_test = _input0