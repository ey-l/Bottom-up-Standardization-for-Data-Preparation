import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostRegressor
from catboost import Pool
from sklearn import svm
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from catboost import CatBoostRegressor
from catboost import Pool
from xgboost import XGBRegressor
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input2 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
_input1.shape
_input1.head(4)
_input1.shape
_input1 = _input1.drop(['Alley', 'PoolQC', 'Fence', 'FireplaceQu', 'LotFrontage', 'GarageYrBlt', 'MiscFeature'], axis=1)
_input0 = _input0.drop(['Alley', 'PoolQC', 'Fence', 'FireplaceQu', 'LotFrontage', 'GarageYrBlt', 'MiscFeature'], axis=1)
_input1.shape
list1 = ['BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond']
list2 = ['BsmtFinType1', 'BsmtFinType2', 'GarageFinish']
for m in list1:
    _input1[m] = _input1[m].fillna('TA')
    _input0[m] = _input0[m].fillna('TA')
for j in list2:
    _input1[j] = _input1[j].fillna('Unf')
    _input0[j] = _input0[j].fillna('Unf')
_input1['BsmtExposure'] = _input1['BsmtExposure'].fillna('NO')
_input1['GarageType'] = _input1['GarageType'].fillna('Attchd')
_input1['MasVnrType'] = _input1['MasVnrType'].fillna('None')
_input1['MasVnrArea'] = _input1['MasVnrArea'].fillna(0)
_input1['Electrical'] = _input1['Electrical'].fillna('SBrkr')
_input0['BsmtExposure'] = _input0['BsmtExposure'].fillna('NO')
_input0['GarageType'] = _input0['GarageType'].fillna('Attchd')
_input0['MasVnrType'] = _input0['MasVnrType'].fillna('None')
_input0['MasVnrArea'] = _input0['MasVnrArea'].fillna(0)
_input0['Electrical'] = _input0['Electrical'].fillna('SBrkr')
_input0['BsmtFinSF1'] = _input0['BsmtFinSF1'].fillna(439)
_input0['BsmtFinSF2'] = _input0['BsmtFinSF2'].fillna(53)
_input0['BsmtUnfSF'] = _input0['BsmtUnfSF'].fillna(554)
_input0['TotalBsmtSF'] = _input0['TotalBsmtSF'].fillna(1046)
_input0['BsmtFullBath'] = _input0['BsmtFullBath'].fillna(1)
_input0['BsmtHalfBath'] = _input0['BsmtHalfBath'].fillna(1)
_input0['GarageCars'] = _input0['GarageCars'].fillna(2)
_input0['GarageArea'] = _input0['GarageArea'].fillna(217)
le = preprocessing.LabelEncoder()
list3 = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']
_input0 = _input0.astype('str')
for i in list3:
    _input1[i] = le.fit_transform(_input1[i])
    _input0[i] = le.fit_transform(_input0[i])
X = _input1.drop('SalePrice', axis=1)
y = _input1.SalePrice
clf = RandomForestRegressor()