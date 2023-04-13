import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectFromModel
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
yTrain = _input1.SalePrice
_input1 = _input1.drop(['SalePrice'], axis=1, inplace=False)
trainAndTest = _input1.append(_input0)
trainAndTest = trainAndTest.reset_index(inplace=False)
trainAndTest = trainAndTest.drop(['index', 'Id'], inplace=False, axis=1)
print(_input1.shape[0] + _input0.shape[0], trainAndTest.shape[0])
trainAndTest
isNA = trainAndTest.isna().sum()[trainAndTest.isna().sum() != 0]
isNA
trainAndTest['MSZoning'] = trainAndTest['MSZoning'].fillna(trainAndTest['MSZoning'].value_counts().idxmax(), inplace=False)
trainAndTest['Alley'] = trainAndTest['Alley'].fillna('NoValue', inplace=False)
trainAndTest['LotFrontage'] = trainAndTest['LotFrontage'].fillna(trainAndTest['LotFrontage'].mean(), inplace=False)
trainAndTest['Utilities'] = trainAndTest['Utilities'].fillna(trainAndTest['Utilities'].value_counts().idxmax(), inplace=False)
trainAndTest['Exterior1st'] = trainAndTest['Exterior1st'].fillna(trainAndTest['Exterior1st'].value_counts().idxmax(), inplace=False)
trainAndTest['Exterior2nd'] = trainAndTest['Exterior2nd'].fillna(trainAndTest['Exterior2nd'].value_counts().idxmax(), inplace=False)
trainAndTest['MasVnrType'] = trainAndTest['MasVnrType'].fillna('NoValue', inplace=False)
trainAndTest['MasVnrArea'] = trainAndTest['MasVnrArea'].fillna(0.0, inplace=False)
trainAndTest['BsmtQual'] = trainAndTest['BsmtQual'].fillna('NoValue', inplace=False)
trainAndTest['BsmtCond'] = trainAndTest['BsmtCond'].fillna('NoValue', inplace=False)
trainAndTest['BsmtExposure'] = trainAndTest['BsmtExposure'].fillna('NoValue', inplace=False)
trainAndTest['BsmtFinType1'] = trainAndTest['BsmtFinType1'].fillna('NoValue', inplace=False)
trainAndTest['BsmtFinSF1'] = trainAndTest['BsmtFinSF1'].fillna(0.0, inplace=False)
trainAndTest['BsmtFinType2'] = trainAndTest['BsmtFinType2'].fillna('NoValue', inplace=False)
trainAndTest['BsmtFinSF2'] = trainAndTest['BsmtFinSF2'].fillna(0.0, inplace=False)
trainAndTest['BsmtUnfSF'] = trainAndTest['BsmtUnfSF'].fillna(0.0, inplace=False)
trainAndTest['TotalBsmtSF'] = trainAndTest['TotalBsmtSF'].fillna(0.0, inplace=False)
trainAndTest['Electrical'] = trainAndTest['Electrical'].fillna(trainAndTest['Electrical'].value_counts().idxmax(), inplace=False)
trainAndTest['BsmtFullBath'] = trainAndTest['BsmtFullBath'].fillna(trainAndTest['BsmtFullBath'].value_counts().idxmax(), inplace=False)
trainAndTest['BsmtHalfBath'] = trainAndTest['BsmtHalfBath'].fillna(trainAndTest['BsmtHalfBath'].value_counts().idxmax(), inplace=False)
trainAndTest['KitchenQual'] = trainAndTest['KitchenQual'].fillna(trainAndTest['KitchenQual'].value_counts().idxmax(), inplace=False)
trainAndTest['Functional'] = trainAndTest['Functional'].fillna(trainAndTest['Functional'].value_counts().idxmax(), inplace=False)
trainAndTest['FireplaceQu'] = trainAndTest['FireplaceQu'].fillna('NoValue', inplace=False)
trainAndTest['GarageType'] = trainAndTest['GarageType'].fillna('NoValue', inplace=False)
trainAndTest['GarageYrBlt'] = trainAndTest['GarageYrBlt'].fillna(0.0, inplace=False)
trainAndTest['GarageFinish'] = trainAndTest['GarageFinish'].fillna('NoValue', inplace=False)
trainAndTest['GarageCars'] = trainAndTest['GarageCars'].fillna(trainAndTest['GarageCars'].value_counts().idxmax(), inplace=False)
trainAndTest['GarageArea'] = trainAndTest['GarageArea'].fillna(trainAndTest['GarageArea'].mean(), inplace=False)
trainAndTest['GarageQual'] = trainAndTest['GarageQual'].fillna('NoValue', inplace=False)
trainAndTest['GarageCond'] = trainAndTest['GarageCond'].fillna('NoValue', inplace=False)
trainAndTest['PoolQC'] = trainAndTest['PoolQC'].fillna('NoValue', inplace=False)
trainAndTest['Fence'] = trainAndTest['Fence'].fillna('NoValue', inplace=False)
trainAndTest['MiscFeature'] = trainAndTest['MiscFeature'].fillna('NoValue', inplace=False)
trainAndTest['SaleType'] = trainAndTest['SaleType'].fillna(trainAndTest['SaleType'].value_counts().idxmax(), inplace=False)
isNA = trainAndTest.isna().sum()[trainAndTest.isna().sum() != 0]
isNA
categoricalVars = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
oneHotEncoder = OneHotEncoder(sparse=False, drop='first')
trainAndTestEncoded = oneHotEncoder.fit_transform(trainAndTest[categoricalVars])
encoderFeatureNames = oneHotEncoder.get_feature_names(categoricalVars)
trainAndTestEncoded = pd.DataFrame(trainAndTestEncoded, columns=encoderFeatureNames)
trainAndTest = pd.concat([trainAndTest.reset_index(drop=True), trainAndTestEncoded.reset_index(drop=True)], axis=1)
trainAndTest = trainAndTest.drop(categoricalVars, axis=1, inplace=False)
trainNRows = _input1.shape[0]
trainFE = trainAndTest.iloc[:trainNRows]
testFE = trainAndTest.iloc[trainNRows:]
print('training set', _input1.shape, trainFE.shape)
print('test set', _input0.shape, testFE.shape)
(X_train, X_test, y_train, y_test) = train_test_split(trainFE, yTrain, test_size=0.2, random_state=42)
regressor = RandomForestRegressor(random_state=42, oob_score=True)