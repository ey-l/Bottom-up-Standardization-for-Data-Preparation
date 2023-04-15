import pandas as pd
import numpy as np
import os
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
import warnings

def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn
house_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
house_data_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train_parent = house_data
test_parent = house_data_test
house_data = house_data.drop('Id', axis=1)
house_data_test = house_data_test.drop('Id', axis=1)
threshold = 0.4 * len(house_data)
df = pd.DataFrame(len(house_data) - house_data.count(), columns=['count'])
df.index[df['count'] > threshold]
house_data = house_data.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)
house_data_test = house_data_test.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)
house_data['SalePrice'].describe()
house_data.select_dtypes(include=np.number).columns
for col in ('MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold'):
    house_data[col] = house_data[col].fillna(0)
    house_data_test[col] = house_data_test[col].fillna('0')
house_data.select_dtypes(exclude=np.number).columns
for col in ('MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition'):
    house_data[col] = house_data[col].fillna('None')
    house_data_test[col] = house_data_test[col].fillna('None')
house_data[house_data.isnull().any(axis=1)]
house_data_test[house_data_test.isnull().any(axis=1)]
train = house_data
test = house_data_test
train['train'] = 1
test['train'] = 0
combined = pd.concat([train, test])
ohe_data_frame = pd.get_dummies(combined, columns=['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition'])
train_df = ohe_data_frame[ohe_data_frame['train'] == 1]
test_df = ohe_data_frame[ohe_data_frame['train'] == 0]
train_df.drop(['train'], axis=1, inplace=True)
test_df.drop(['train', 'SalePrice'], axis=1, inplace=True)
house_data = train_df
house_data_test = test_df
X_train = house_data.drop('SalePrice', axis=1)
Y_train = house_data['SalePrice']
X_test = house_data_test