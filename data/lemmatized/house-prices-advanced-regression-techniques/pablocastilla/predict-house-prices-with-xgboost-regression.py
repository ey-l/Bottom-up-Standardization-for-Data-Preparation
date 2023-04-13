import numpy as np
import pandas as pd
import xgboost
import csv as csv
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix, mean_squared_error, precision_score
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', header=0)
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', header=0)
categorical_features = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
_input1.describe()
features_with_nan = ['Alley', 'MasVnrType', 'BsmtQual', 'BsmtQual', 'BsmtCond', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish']

def ConverNaNToNAString(data, columnList):
    for x in columnList:
        data[x] = str(data[x])
ConverNaNToNAString(_input1, features_with_nan)
ConverNaNToNAString(_input0, features_with_nan)

def CreateColumnPerValue(data, columnList):
    for x in columnList:
        values = pd.unique(data[x])
        for v in values:
            column_name = x + '_' + str(v)
            data[column_name] = (data[x] == v).astype(float)
        data = data.drop(x, axis=1, inplace=False)
CreateColumnPerValue(_input1, categorical_features)
CreateColumnPerValue(_input0, categorical_features)
y = _input1['SalePrice']
_input1 = _input1.drop('SalePrice', axis=1, inplace=False)
model = xgboost.XGBRegressor()