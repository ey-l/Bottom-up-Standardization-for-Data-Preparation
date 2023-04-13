import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1.head()
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input2 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
df = pd.concat([_input1, _input0], axis=0, sort=False)
df.shape
df = df.set_index('Id')
train_id = _input1.shape[0]
SEED = 66
for i in df.columns:
    if df[i].isna().sum() > 0:
        print(i + ' : ' + str(round(df[i].isna().sum() / df.shape[0] * 100, 2)) + ' %')
to_fill_none = ['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature', 'Exterior2nd']
to_fill_0 = ['LotFrontage', 'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageArea', 'BsmtFinSF1', 'BsmtFinSF2', 'GarageCars']
to_fill_freq = ['GarageYrBlt', 'Electrical', 'MSZoning', 'Utilities', 'KitchenQual', 'Functional', 'Exterior1st', 'SaleType']
for i in to_fill_none:
    df[i] = df[i].fillna('None', inplace=False)
for i in to_fill_0:
    df[i] = df[i].fillna(0, inplace=False)
for i in to_fill_freq:
    df[i] = df[i].fillna(df[i][:train_id].mode().item(), inplace=False)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

def encoder(dataset, encoder=LabelEncoder()):
    var_obj = dataset.select_dtypes(include='object').columns
    dataset_cop = dataset.copy()
    le = encoder
    for i in var_obj:
        dataset_cop[i] = le.fit_transform(dataset_cop[i])
    return dataset_cop

def get_data(dataset):
    dataset = encoder(dataset)
    _input0 = dataset[dataset.index > train_id].copy()
    X = dataset.loc[set(dataset.index) - set(_input0.index)].copy()
    y = X.pop('SalePrice')
    return (X, y, _input0)

def split(X, y, seed=SEED):
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=seed)
    return (X_train, X_test, y_train, y_test)

def scaler(X_train, X_test, scaler_func, df_test=None):
    scal = scaler_func
    X_train = scal.fit_transform(X_train)
    X_test = scal.transform(X_test)
    if _input0 is not None:
        _input0 = _input0.drop('SalePrice', axis=1, inplace=False)
        _input0 = scal.transform(_input0)
    return (X_train, X_test, _input0)

def get_score(dataset, estimator, scaler_func=StandardScaler()):
    dataset = encoder(dataset)
    (X, y, _) = get_data(dataset)
    (X_train, X_test, y_train, y_test) = split(X, y)
    (X_train, X_test, _) = scaler(X_train, X_test, scaler_func)