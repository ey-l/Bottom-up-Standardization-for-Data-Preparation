import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1
_input1 = _input1.set_index('Id', inplace=False)
_input1.info()
len(_input1)
for col in _input1.columns:
    if _input1[col].isnull().sum() > 0:
        print('%s has %d null values' % (col, _input1[col].isnull().sum()))
more_than_1000 = ['Alley', 'PoolQC', 'Fence', 'MiscFeature']
_input1 = _input1.drop(more_than_1000, axis=1, inplace=False)
_input0 = _input0.drop(more_than_1000, axis=1, inplace=False)
_input1[_input1['GarageType'].isna()][['GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond']].isna().sum()
garage = ['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond']
for g in garage:
    _input1[g] = _input1[g].fillna('NA')
    _input0[g] = _input0[g].fillna('NA')
_input1['GarageYrBlt'] = _input1['GarageYrBlt'].astype(str)
_input0['GarageYrBlt'] = _input0['GarageYrBlt'].astype(str)
_input1.corr()['LotFrontage'].sort_values(ascending=False)[1:5]
_input1['MasVnrType'] = _input1['MasVnrType'].fillna('None')
_input1['MasVnrArea'] = _input1['MasVnrArea'].fillna(0)
_input0['MasVnrType'] = _input0['MasVnrType'].fillna('None')
_input0['MasVnrArea'] = _input0['MasVnrArea'].fillna(0)
_input1['Electrical'] = _input1['Electrical'].fillna('Mix')
_input0['Electrical'] = _input0['Electrical'].fillna('Mix')
_input1[_input1['BsmtQual'].isnull()][['BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']].isna().sum()
for c in range(len(_input1.columns)):
    if 'Bsmt' in _input1.columns[c]:
        print('%s-->%d' % (_input1.columns[c], c))
_input1[~_input1['BsmtQual'].isnull() & _input1['BsmtExposure'].isnull()]
_input1.iat[948, 31] = 'No'
_input1[~_input1['BsmtQual'].isnull() & _input1['BsmtFinType2'].isnull()]
_input1.iat[332, 34] = 'Unf'
basement = ['Qual', 'Cond', 'Exposure', 'FinType1', 'FinType2']
for b in basement:
    _input1['Bsmt' + b] = _input1['Bsmt' + b].fillna('NA')
    _input0['Bsmt' + b] = _input0['Bsmt' + b].fillna('NA')
_input0.isna().sum()[_input0.isna().sum() > 0]
_input0 = _input0.set_index('Id', inplace=False)
_input0['MSZoning'] = _input0['MSZoning'].fillna(_input0['MSZoning'].mode())
_input0['Utilities'] = _input0['Utilities'].fillna('AllPub')
_input0['Exterior1st'] = _input0['Exterior1st'].fillna('VinylSd')
_input0['Exterior2nd'] = _input0['Exterior2nd'].fillna('VinylSd')
_input0[_input0['BsmtFinSF1'].isna()][['BsmtUnfSF', 'BsmtFinSF2', 'TotalBsmtSF', 'BsmtCond', 'BsmtFullBath']]
_input0[_input0['BsmtFullBath'].isna()].iloc[:, 45:47].index
_input0.at[2189, 'BsmtHalfBath'] = 0
_input0.at[2189, 'BsmtFullBath'] = 0
_input0['BsmtFinSF1'] = _input0['BsmtFinSF1'].fillna(0)
_input0['BsmtFinSF2'] = _input0['BsmtFinSF2'].fillna(0)
_input0['BsmtUnfSF'] = _input0['BsmtUnfSF'].fillna(0)
_input0['TotalBsmtSF'] = _input0['TotalBsmtSF'].fillna(0)
_input0['BsmtFullBath'] = _input0['BsmtFullBath'].fillna(0)
_input0['BsmtHalfBath'] = _input0['BsmtHalfBath'].fillna(0)
_input0['KitchenQual'] = _input0['KitchenQual'].fillna(_input1['KitchenQual'].mode().loc[0])
_input0['Functional'] = _input0['Functional'].fillna(_input1['Functional'].mode().loc[0])
_input0['GarageCars'] = _input0['GarageCars'].fillna(_input1['GarageCars'].mode().loc[0])
_input0['GarageArea'] = _input0['GarageArea'].fillna(_input1['GarageArea'].mean())
_input0['SaleType'] = _input0['SaleType'].fillna(_input1['SaleType'].mode().loc[0])
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
cols = ['1stFlrSF', 'LotArea', 'GrLivArea', 'TotalBsmtSF', 'LotFrontage']
lotfront_df = _input1[cols].copy()
null_lotfront_df = lotfront_df[lotfront_df['LotFrontage'].isnull()].copy()
lotfront_df = lotfront_df.dropna(inplace=False)
X = lotfront_df.drop('LotFrontage', axis=1)
y = lotfront_df['LotFrontage']
sc = StandardScaler()
X_sc = sc.fit_transform(X)
lr = LinearRegression()