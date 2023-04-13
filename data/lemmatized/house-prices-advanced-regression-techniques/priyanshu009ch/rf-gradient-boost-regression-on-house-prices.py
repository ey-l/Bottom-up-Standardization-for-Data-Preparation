import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1.head()
for col in _input1.columns:
    if _input1[col].count() < 1460:
        print(col)
m = _input1['LotFrontage'].mean()
_input1['LotFrontage'] = _input1['LotFrontage'].fillna(value=m, inplace=False)
_input1['Alley'] = _input1['Alley'].fillna(value='None', inplace=False)
_input1['MasVnrType'] = _input1['MasVnrType'].fillna(value='None', inplace=False)
m = _input1['MasVnrArea'].mean()
_input1['MasVnrArea'] = _input1['MasVnrArea'].fillna(value=m, inplace=False)
_input1['BsmtQual'] = _input1['BsmtQual'].fillna(value='None', inplace=False)
_input1['BsmtCond'] = _input1['BsmtCond'].fillna(value='None', inplace=False)
_input1['BsmtExposure'] = _input1['BsmtExposure'].fillna(value='None', inplace=False)
_input1['BsmtFinType1'] = _input1['BsmtFinType1'].fillna(value='None', inplace=False)
_input1['BsmtFinType2'] = _input1['BsmtFinType2'].fillna(value='None', inplace=False)
_input1['Electrical'] = _input1['Electrical'].fillna(value='None', inplace=False)
_input1['FireplaceQu'] = _input1['FireplaceQu'].fillna(value='None', inplace=False)
_input1['GarageType'] = _input1['GarageType'].fillna(value='None', inplace=False)
m = _input1['GarageYrBlt'].mean()
_input1['GarageYrBlt'] = _input1['GarageYrBlt'].fillna(value=m, inplace=False)
_input1['GarageFinish'] = _input1['GarageFinish'].fillna(value='None', inplace=False)
_input1['GarageQual'] = _input1['GarageQual'].fillna(value='None', inplace=False)
_input1['GarageCond'] = _input1['GarageCond'].fillna(value='None', inplace=False)
_input1['PoolQC'] = _input1['PoolQC'].fillna(value='None', inplace=False)
_input1['Fence'] = _input1['Fence'].fillna(value='None', inplace=False)
_input1['MiscFeature'] = _input1['MiscFeature'].fillna(value='None', inplace=False)
for col in _input1.columns:
    if _input1[col].count() < 1460:
        print(col)
for col in _input1.columns:
    if type(_input1[col][0]) == str:
        _input1[col] = _input1[col].astype('category')
        _input1[col] = _input1[col].cat.codes
_input1.head()
x = _input1.iloc[:, 1:-1].values
y = _input1.iloc[:, -1].values
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2, random_state=0)
print(len(x), len(x_train), len(x_test))
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=0)