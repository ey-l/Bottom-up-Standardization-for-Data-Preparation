import numpy as np
import pandas as pd
import xgboost
import csv as csv
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix, mean_squared_error, precision_score
train_dataset = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', header=0)
test_dataset = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', header=0)
categorical_features = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
train_dataset.describe()
features_with_nan = ['Alley', 'MasVnrType', 'BsmtQual', 'BsmtQual', 'BsmtCond', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish']

def ConverNaNToNAString(data, columnList):
    for x in columnList:
        data[x] = str(data[x])
ConverNaNToNAString(train_dataset, features_with_nan)
ConverNaNToNAString(test_dataset, features_with_nan)

def CreateColumnPerValue(data, columnList):
    for x in columnList:
        values = pd.unique(data[x])
        for v in values:
            column_name = x + '_' + str(v)
            data[column_name] = (data[x] == v).astype(float)
        data.drop(x, axis=1, inplace=True)
CreateColumnPerValue(train_dataset, categorical_features)
CreateColumnPerValue(test_dataset, categorical_features)
y = train_dataset['SalePrice']
train_dataset.drop('SalePrice', axis=1, inplace=True)
model = xgboost.XGBRegressor()