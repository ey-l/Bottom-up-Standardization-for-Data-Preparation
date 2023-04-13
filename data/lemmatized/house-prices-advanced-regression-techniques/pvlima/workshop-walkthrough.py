import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_numeric_dtype
sns.set()
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print('Train shape:', _input1.shape)
print('Test shape:', _input0.shape)
_input1.dtypes.value_counts()
selected = ['GrLivArea', 'LotArea', 'BsmtUnfSF', '1stFlrSF', 'TotalBsmtSF', 'GarageArea', 'BsmtFinSF1', 'LotFrontage', 'YearBuilt', 'Neighborhood', 'GarageYrBlt', 'OpenPorchSF', 'YearRemodAdd', 'WoodDeckSF', 'MoSold', '2ndFlrSF', 'OverallCond', 'Exterior1st', 'YrSold', 'OverallQual']
train = _input1[selected].copy()
train['is_train'] = 1
train['SalePrice'] = _input1['SalePrice'].values
train['Id'] = _input1['Id'].values
test = _input0[selected].copy()
test['is_train'] = 0
test['SalePrice'] = 1
test['Id'] = _input0['Id'].values
full = pd.concat([train, test])
not_features = ['Id', 'SalePrice', 'is_train']
features = [c for c in train.columns if c not in not_features]
pd.Series(train.SalePrice).hist(bins=50)
pd.Series(np.log(train.SalePrice)).hist(bins=50)
full['SalePrice'] = np.log(full['SalePrice'])

def summary(df, dtype):
    data = []
    for c in df.select_dtypes([dtype]).columns:
        data.append({'name': c, 'unique': df[c].nunique(), 'nulls': df[c].isnull().sum(), 'samples': df[c].unique()[:20]})
    return pd.DataFrame(data)
summary(full[features], np.object)
summary(full[features], np.float64)
summary(full[features], np.int64)
for c in full.select_dtypes([np.object]).columns:
    full[c] = full[c].fillna('__NA__', inplace=False)
for c in full.select_dtypes([np.float64]).columns:
    full[c] = full[c].fillna(0, inplace=False)
for c in full.columns:
    assert full[c].isnull().sum() == 0, f'There are still missing values in {c}'
mappers = {}
for c in full.select_dtypes([np.object]).columns:
    mappers[c] = {v: i for (i, v) in enumerate(full[c].unique())}
    full[c] = full[c].map(mappers[c]).astype(int)
for c in full.columns:
    assert is_numeric_dtype(full[c]), f'Non-numeric column {c}'
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics

def rmse(y_true, y_pred):
    return np.sqrt(metrics.mean_squared_error(y_true, y_pred))
train = full[full.is_train == 1][features].values
target = full[full.is_train == 1].SalePrice.values
(Xtrain, Xvalid, ytrain, yvalid) = train_test_split(train, target, test_size=0.2, random_state=42)
model = GradientBoostingRegressor(n_estimators=1500, learning_rate=0.02, max_depth=4, random_state=42)