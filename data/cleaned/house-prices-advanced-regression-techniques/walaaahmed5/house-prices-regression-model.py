import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print('train data : {}'.format(train.shape))
print('test data : {}'.format(test.shape))
train_id = train['Id']
test_id = test['Id']
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train['SalePrice']
train.drop(['SalePrice'], axis=1, inplace=True)
all_data = pd.concat((train, test), ignore_index=True)
print('all_data size is : {}'.format(all_data.shape))
all_data.head()
all_data.info()
all_data.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu'], axis=1, inplace=True)
all_data.shape
print(all_data[['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']].head())
all_data.drop(['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF'], axis=1, inplace=True)
all_data.shape
categoral_mode = ['MSZoning', 'MasVnrType', 'Electrical', 'SaleType', 'Utilities', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'Functional']
for col in categoral_mode:
    all_data[col].fillna(all_data[col].mode()[0], inplace=True)
no_basement = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
for col in no_basement:
    all_data[col].fillna('NoB', inplace=True)
no_garage = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
for col in no_garage:
    all_data[col].fillna('NoG', inplace=True)
numric_mean = ['LotFrontage', 'GarageYrBlt']
for col in numric_mean:
    all_data[col].fillna(all_data[col].median(), inplace=True)
numric_zero = ['MasVnrArea', 'GarageCars', 'GarageArea', 'BsmtFullBath', 'BsmtHalfBath', 'TotalBsmtSF']
for col in numric_zero:
    all_data[col].fillna(0, inplace=True)
all_data.info()
one_hot = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'Functional', 'GarageType', 'SaleType', 'SaleCondition']
all_data = pd.get_dummies(all_data, columns=one_hot)
ordinal = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive']
ordinal_mapping = {'ExterQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'ExterCond': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'BsmtQual': {'NoB': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'BsmtCond': {'NoB': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'BsmtExposure': {'NoB': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}, 'BsmtFinType1': {'NoB': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}, 'BsmtFinType2': {'NoB': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}, 'HeatingQC': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'CentralAir': {'N': 0, 'Y': 1}, 'Electrical': {'Mix': 1, 'FuseP': 2, 'FuseF': 3, 'FuseA': 4, 'SBrkr': 5}, 'KitchenQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'GarageFinish': {'NoG': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}, 'GarageQual': {'NoG': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'GarageCond': {'NoG': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'PavedDrive': {'N': 1, 'P': 2, 'Y': 3}}
all_data.replace(ordinal_mapping, inplace=True)
all_data.info()
all_data.head()
x_train = all_data[:ntrain]
x_test = all_data[ntrain:]
from sklearn.model_selection import train_test_split
(X_train, X_valid, y_train, y_valid) = train_test_split(x_train, y_train, train_size=0.8, test_size=0.2, random_state=0)
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
final_GBR_model = GradientBoostingRegressor(learning_rate=0.01, n_estimators=1000, max_depth=4)