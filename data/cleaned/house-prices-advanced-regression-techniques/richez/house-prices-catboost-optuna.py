import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df_train.head()
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
sub = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
df = pd.concat([df_train, df_test], axis=0, sort=False)
df.shape
df = df.set_index('Id')
train_id = df_train.shape[0]
SEED = 66
for i in df.columns:
    if df[i].isna().sum() > 0:
        print(i + ' : ' + str(round(df[i].isna().sum() / df.shape[0] * 100, 2)) + ' %')
to_fill_none = ['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature', 'Exterior2nd']
to_fill_0 = ['LotFrontage', 'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageArea', 'BsmtFinSF1', 'BsmtFinSF2', 'GarageCars']
to_fill_freq = ['GarageYrBlt', 'Electrical', 'MSZoning', 'Utilities', 'KitchenQual', 'Functional', 'Exterior1st', 'SaleType']
for i in to_fill_none:
    df[i].fillna('None', inplace=True)
for i in to_fill_0:
    df[i].fillna(0, inplace=True)
for i in to_fill_freq:
    df[i].fillna(df[i][:train_id].mode().item(), inplace=True)
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
    df_test = dataset[dataset.index > train_id].copy()
    X = dataset.loc[set(dataset.index) - set(df_test.index)].copy()
    y = X.pop('SalePrice')
    return (X, y, df_test)

def split(X, y, seed=SEED):
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=seed)
    return (X_train, X_test, y_train, y_test)

def scaler(X_train, X_test, scaler_func, df_test=None):
    scal = scaler_func
    X_train = scal.fit_transform(X_train)
    X_test = scal.transform(X_test)
    if df_test is not None:
        df_test.drop('SalePrice', axis=1, inplace=True)
        df_test = scal.transform(df_test)
    return (X_train, X_test, df_test)

def get_score(dataset, estimator, scaler_func=StandardScaler()):
    dataset = encoder(dataset)
    (X, y, _) = get_data(dataset)
    (X_train, X_test, y_train, y_test) = split(X, y)
    (X_train, X_test, _) = scaler(X_train, X_test, scaler_func)