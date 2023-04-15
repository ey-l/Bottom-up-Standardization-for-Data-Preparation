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
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
yTrain = train.SalePrice
train.drop(['SalePrice'], axis=1, inplace=True)
trainAndTest = train.append(test)
trainAndTest.reset_index(inplace=True)
trainAndTest.drop(['index', 'Id'], inplace=True, axis=1)
print(train.shape[0] + test.shape[0], trainAndTest.shape[0])
trainAndTest
isNA = trainAndTest.isna().sum()[trainAndTest.isna().sum() != 0]
isNA
trainAndTest['MSZoning'].fillna(trainAndTest['MSZoning'].value_counts().idxmax(), inplace=True)
trainAndTest['Alley'].fillna('NoValue', inplace=True)
trainAndTest['LotFrontage'].fillna(trainAndTest['LotFrontage'].mean(), inplace=True)
trainAndTest['Utilities'].fillna(trainAndTest['Utilities'].value_counts().idxmax(), inplace=True)
trainAndTest['Exterior1st'].fillna(trainAndTest['Exterior1st'].value_counts().idxmax(), inplace=True)
trainAndTest['Exterior2nd'].fillna(trainAndTest['Exterior2nd'].value_counts().idxmax(), inplace=True)
trainAndTest['MasVnrType'].fillna('NoValue', inplace=True)
trainAndTest['MasVnrArea'].fillna(0.0, inplace=True)
trainAndTest['BsmtQual'].fillna('NoValue', inplace=True)
trainAndTest['BsmtCond'].fillna('NoValue', inplace=True)
trainAndTest['BsmtExposure'].fillna('NoValue', inplace=True)
trainAndTest['BsmtFinType1'].fillna('NoValue', inplace=True)
trainAndTest['BsmtFinSF1'].fillna(0.0, inplace=True)
trainAndTest['BsmtFinType2'].fillna('NoValue', inplace=True)
trainAndTest['BsmtFinSF2'].fillna(0.0, inplace=True)
trainAndTest['BsmtUnfSF'].fillna(0.0, inplace=True)
trainAndTest['TotalBsmtSF'].fillna(0.0, inplace=True)
trainAndTest['Electrical'].fillna(trainAndTest['Electrical'].value_counts().idxmax(), inplace=True)
trainAndTest['BsmtFullBath'].fillna(trainAndTest['BsmtFullBath'].value_counts().idxmax(), inplace=True)
trainAndTest['BsmtHalfBath'].fillna(trainAndTest['BsmtHalfBath'].value_counts().idxmax(), inplace=True)
trainAndTest['KitchenQual'].fillna(trainAndTest['KitchenQual'].value_counts().idxmax(), inplace=True)
trainAndTest['Functional'].fillna(trainAndTest['Functional'].value_counts().idxmax(), inplace=True)
trainAndTest['FireplaceQu'].fillna('NoValue', inplace=True)
trainAndTest['GarageType'].fillna('NoValue', inplace=True)
trainAndTest['GarageYrBlt'].fillna(0.0, inplace=True)
trainAndTest['GarageFinish'].fillna('NoValue', inplace=True)
trainAndTest['GarageCars'].fillna(trainAndTest['GarageCars'].value_counts().idxmax(), inplace=True)
trainAndTest['GarageArea'].fillna(trainAndTest['GarageArea'].mean(), inplace=True)
trainAndTest['GarageQual'].fillna('NoValue', inplace=True)
trainAndTest['GarageCond'].fillna('NoValue', inplace=True)
trainAndTest['PoolQC'].fillna('NoValue', inplace=True)
trainAndTest['Fence'].fillna('NoValue', inplace=True)
trainAndTest['MiscFeature'].fillna('NoValue', inplace=True)
trainAndTest['SaleType'].fillna(trainAndTest['SaleType'].value_counts().idxmax(), inplace=True)
isNA = trainAndTest.isna().sum()[trainAndTest.isna().sum() != 0]
isNA
categoricalVars = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
oneHotEncoder = OneHotEncoder(sparse=False, drop='first')
trainAndTestEncoded = oneHotEncoder.fit_transform(trainAndTest[categoricalVars])
encoderFeatureNames = oneHotEncoder.get_feature_names(categoricalVars)
trainAndTestEncoded = pd.DataFrame(trainAndTestEncoded, columns=encoderFeatureNames)
trainAndTest = pd.concat([trainAndTest.reset_index(drop=True), trainAndTestEncoded.reset_index(drop=True)], axis=1)
trainAndTest.drop(categoricalVars, axis=1, inplace=True)
trainNRows = train.shape[0]
trainFE = trainAndTest.iloc[:trainNRows]
testFE = trainAndTest.iloc[trainNRows:]
print('training set', train.shape, trainFE.shape)
print('test set', test.shape, testFE.shape)
(X_train, X_test, y_train, y_test) = train_test_split(trainFE, yTrain, test_size=0.2, random_state=42)
regressor = RandomForestRegressor(random_state=42, oob_score=True)