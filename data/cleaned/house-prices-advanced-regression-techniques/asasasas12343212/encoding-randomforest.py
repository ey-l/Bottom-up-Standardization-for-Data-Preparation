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
data_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
data_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
data_sub = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
data_train.shape
data_train.head(4)
data_train.shape
data_train = data_train.drop(['Alley', 'PoolQC', 'Fence', 'FireplaceQu', 'LotFrontage', 'GarageYrBlt', 'MiscFeature'], axis=1)
data_test = data_test.drop(['Alley', 'PoolQC', 'Fence', 'FireplaceQu', 'LotFrontage', 'GarageYrBlt', 'MiscFeature'], axis=1)
data_train.shape
list1 = ['BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond']
list2 = ['BsmtFinType1', 'BsmtFinType2', 'GarageFinish']
for m in list1:
    data_train[m] = data_train[m].fillna('TA')
    data_test[m] = data_test[m].fillna('TA')
for j in list2:
    data_train[j] = data_train[j].fillna('Unf')
    data_test[j] = data_test[j].fillna('Unf')
data_train['BsmtExposure'] = data_train['BsmtExposure'].fillna('NO')
data_train['GarageType'] = data_train['GarageType'].fillna('Attchd')
data_train['MasVnrType'] = data_train['MasVnrType'].fillna('None')
data_train['MasVnrArea'] = data_train['MasVnrArea'].fillna(0)
data_train['Electrical'] = data_train['Electrical'].fillna('SBrkr')
data_test['BsmtExposure'] = data_test['BsmtExposure'].fillna('NO')
data_test['GarageType'] = data_test['GarageType'].fillna('Attchd')
data_test['MasVnrType'] = data_test['MasVnrType'].fillna('None')
data_test['MasVnrArea'] = data_test['MasVnrArea'].fillna(0)
data_test['Electrical'] = data_test['Electrical'].fillna('SBrkr')
data_test['BsmtFinSF1'] = data_test['BsmtFinSF1'].fillna(439)
data_test['BsmtFinSF2'] = data_test['BsmtFinSF2'].fillna(53)
data_test['BsmtUnfSF'] = data_test['BsmtUnfSF'].fillna(554)
data_test['TotalBsmtSF'] = data_test['TotalBsmtSF'].fillna(1046)
data_test['BsmtFullBath'] = data_test['BsmtFullBath'].fillna(1)
data_test['BsmtHalfBath'] = data_test['BsmtHalfBath'].fillna(1)
data_test['GarageCars'] = data_test['GarageCars'].fillna(2)
data_test['GarageArea'] = data_test['GarageArea'].fillna(217)
le = preprocessing.LabelEncoder()
list3 = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']
data_test = data_test.astype('str')
for i in list3:
    data_train[i] = le.fit_transform(data_train[i])
    data_test[i] = le.fit_transform(data_test[i])
X = data_train.drop('SalePrice', axis=1)
y = data_train.SalePrice
clf = RandomForestRegressor()