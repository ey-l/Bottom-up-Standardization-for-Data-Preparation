import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input0
(_input1.shape, _input0.shape)
_input1.info()
print()
print('------------------------------------------------------------')
print('---------------------- Training set ------------------------')
print('------------------------------------------------------------')
print()
count = 1
for i in _input1.columns:
    if _input1[i].isnull().sum() > 0:
        if _input1[i].dtypes == 'object':
            print(i + ' is a Categorical Variable')
            print('Total null values:', _input1[i].isnull().sum())
            print('Null values as a % of total:', round(_input1[i].isnull().sum() * 100 / _input1['SalePrice'].count(), 1))
            print('Categorical Variable No: ' + str(count))
            count = count + 1
            print()
print()
print('------------------------------------------------------------')
print('------------------------ -------- --------------------------')
print('------------------------------------------------------------')
print()
print()
print('------------------------------------------------------------')
print('------------------------ Test set --------------------------')
print('------------------------------------------------------------')
print()
testcount = 1
for i in _input0.columns:
    if _input0[i].isnull().sum() > 0:
        if _input0[i].dtypes == 'object':
            print(i + ' is a Categorical Variable')
            print('Total null values:', _input0[i].isnull().sum())
            print('Null values as a % of total:', round(_input0[i].isnull().sum() * 100 / _input1['SalePrice'].count(), 1))
            print('Categorical Variable No: ' + str(testcount))
            testcount = testcount + 1
            print()
print()
print('------------------------------------------------------------')
print('------------------------ -------- --------------------------')
print('------------------------------------------------------------')
print()
_input1 = _input1.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)
_input0 = _input0.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)
(_input1.shape, _input0.shape)
_input1 = _input1.apply(lambda x: x.fillna(x.value_counts().index[0]))
_input0 = _input0.apply(lambda x: x.fillna(x.value_counts().index[0]))
cats = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
le = LabelEncoder()
for i in _input1[cats]:
    if _input1[i].dtype:
        _input1[i] = le.fit_transform(_input1[i].values)
        print(i + ' has been label encoded')
for i in _input0[cats]:
    if _input0[i].dtype:
        _input0[i] = le.fit_transform(_input0[i].values)
        print(i + ' has been label encoded')
(_input1.shape, _input0.shape)
y_train = _input1['SalePrice']
y_test = _input1['SalePrice']
features = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'GarageQual', 'LandSlope', 'Neighborhood', 'SaleType', 'SaleCondition']
X_train = pd.get_dummies(_input1[features])
X_test = pd.get_dummies(_input0[features])
(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
sc_X = MinMaxScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
model = linear_model.Lasso(alpha=0.1)