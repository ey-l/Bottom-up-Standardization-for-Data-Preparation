import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost
from numpy import *
from scipy import stats
from scipy.interpolate import *
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1.head()
_input1 = _input1.drop(['PoolQC', 'Alley', 'Fence', 'MiscFeature', 'GarageYrBlt'], axis=1, inplace=False)
_input1 = _input1.fillna(_input1.mean(), inplace=False)
_input1['FireplaceQu'] = _input1['FireplaceQu'].fillna(_input1['FireplaceQu'].mode()[0])
_input1['GarageType'] = _input1['GarageType'].fillna(_input1['GarageType'].mode()[0])
_input1['BsmtQual'] = _input1['BsmtQual'].fillna(_input1['BsmtQual'].mode()[0])
_input1['BsmtCond'] = _input1['BsmtCond'].fillna(_input1['BsmtCond'].mode()[0])
_input1['GarageFinish'] = _input1['GarageFinish'].fillna(_input1['GarageFinish'].mode()[0])
_input1['GarageQual'] = _input1['GarageQual'].fillna(_input1['GarageQual'].mode()[0])
_input1['GarageCond'] = _input1['GarageCond'].fillna(_input1['GarageCond'].mode()[0])
_input1['MasVnrType'] = _input1['MasVnrType'].fillna(_input1['MasVnrType'].mode()[0])
_input1['MasVnrArea'] = _input1['MasVnrArea'].fillna(_input1['MasVnrArea'].mode()[0])
_input1['BsmtExposure'] = _input1['BsmtExposure'].fillna(_input1['BsmtExposure'].mode()[0])
_input1['BsmtFinType2'] = _input1['BsmtFinType2'].fillna(_input1['BsmtFinType2'].mode()[0])
_input1 = _input1.dropna(inplace=False)
_input1 = _input1.drop(['Id'], axis=1, inplace=False)
columns = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleCondition', 'SaleType']
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input0 = _input0.drop(['PoolQC', 'Fence', 'Alley', 'MiscFeature', 'GarageYrBlt'], axis=1, inplace=False)
_input0 = _input0.drop(['Id'], axis=1, inplace=False)
_input0 = _input0.fillna(_input0.mean(), inplace=False)
_input0['FireplaceQu'] = _input0['FireplaceQu'].fillna(_input0['FireplaceQu'].mode()[0])
_input0['GarageType'] = _input0['GarageType'].fillna(_input0['GarageType'].mode()[0])
_input0['MasVnrType'] = _input0['MasVnrType'].fillna(_input0['MasVnrType'].mode()[0])
_input0['MasVnrArea'] = _input0['MasVnrArea'].fillna(_input0['MasVnrArea'].mode()[0])
_input0['BsmtQual'] = _input0['BsmtQual'].fillna(_input0['BsmtQual'].mode()[0])
_input0['BsmtCond'] = _input0['BsmtCond'].fillna(_input0['BsmtCond'].mode()[0])
_input0['BsmtExposure'] = _input0['BsmtExposure'].fillna(_input0['BsmtExposure'].mode()[0])
_input0['BsmtFinType1'] = _input0['BsmtFinType1'].fillna(_input0['BsmtFinType1'].mode()[0])
_input0['BsmtFinType2'] = _input0['BsmtFinType2'].fillna(_input0['BsmtFinType2'].mode()[0])
_input0['GarageFinish'] = _input0['GarageFinish'].fillna(_input0['GarageFinish'].mode()[0])
_input0['GarageQual'] = _input0['GarageQual'].fillna(_input0['GarageQual'].mode()[0])
_input0['GarageCond'] = _input0['GarageCond'].fillna(_input0['GarageCond'].mode()[0])
_input0['MSZoning'] = _input0['MSZoning'].fillna(_input0['MSZoning'].mode()[0])
_input0['Utilities'] = _input0['Utilities'].fillna(_input0['Utilities'].mode()[0])
_input0['Exterior1st'] = _input0['Exterior1st'].fillna(_input0['Exterior1st'].mode()[0])
_input0['Exterior2nd'] = _input0['Exterior2nd'].fillna(_input0['Exterior2nd'].mode()[0])
_input0['KitchenQual'] = _input0['KitchenQual'].fillna(_input0['KitchenQual'].mode()[0])
_input0['Functional'] = _input0['Functional'].fillna(_input0['Functional'].mode()[0])
_input0['SaleType'] = _input0['SaleType'].fillna(_input0['SaleType'].mode()[0])
_input0['BsmtFullBath'] = _input0['BsmtFullBath'].fillna(_input0['BsmtFullBath'].mode()[0])
_input0['BsmtHalfBath'] = _input0['BsmtHalfBath'].fillna(_input0['BsmtHalfBath'].mode()[0])

def onehot(multcolumns):
    dffinal = finaldf
    i = 0
    for fields in multcolumns:
        print(fields)
        df1 = pd.get_dummies(finaldf[fields], drop_first=True)
        finaldf = finaldf.drop([fields], axis=1, inplace=False)
        if i == 0:
            dffinal = df1.copy()
        else:
            dffinal = pd.concat([dffinal, df1], axis=1)
        i = i + 1
    dffinal = pd.concat([finaldf, dffinal], axis=1)
    return dffinal
finaldf = pd.concat([_input1, _input0], axis=0)
finaldf = onehot(columns)
finaldf.shape
finaldf = finaldf.loc[:, ~finaldf.columns.duplicated()]
dftrain = finaldf.iloc[:1422, :]
dftest = finaldf.iloc[1422:, :]
dftest = dftest.drop(['SalePrice'], axis=1, inplace=False)
mse = mean_squared_error
Xtrain = dftrain.drop(['SalePrice'], axis=1)
Ytrain = dftrain['SalePrice']
booster = ['gbtree', 'gblinear']
base_score = [0.25, 0.5, 0.75, 1]
n_estimators = [100, 500, 900, 1100, 1300, 1500]
max_depth = [2, 3, 5, 9, 11, 15]
learning_rate = [0.05, 1, 0.15, 0.2]
min_child_weight = [1, 2, 3, 4]
hyperparameter_grid = {'n_estimators': n_estimators, 'max_depth': max_depth, 'learning_rate': learning_rate, 'min_child_weight': min_child_weight, 'booster': booster, 'base_score': base_score}
xg = xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1, gamma=0, importance_type='gain', learning_rate=0.05, max_delta_step=0, max_depth=2, min_child_weight=4, missing=None, n_estimators=1100, n_jobs=1, nthread=None, objective='reg:linear', random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None, silent=None, subsample=1, verbosity=1)