import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import sklearn.metrics as metrics
import math
train_path = '_data/input/house-prices-advanced-regression-techniques/train.csv'
test_path = '_data/input/house-prices-advanced-regression-techniques/test.csv'
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
c_train = train.copy()
c_test = test.copy()
train.shape
train.head()
c_train['train'] = 1
c_test['train'] = 0
df = pd.concat([c_train, c_test], axis=0, sort=False)
df.shape
df
NAN = [(k, df[k].isnull().mean() * 100) for k in df]
print(NAN)
NAN = pd.DataFrame(NAN, columns=['column_name', 'percentage'])
NAN = NAN[NAN.percentage > 50]
NAN.sort_values('percentage', ascending=False)
df = df.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1)
cat_columns = df.select_dtypes(include=['O'])
num_columns = df.select_dtypes(exclude=['O'])
null_cat_columns = cat_columns.isnull().sum()
null_cat_columns
columns_None = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'GarageType', 'GarageFinish', 'GarageQual', 'FireplaceQu', 'GarageCond']
cat_columns[columns_None] = cat_columns[columns_None].fillna('None')
mode_cols = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Electrical', 'KitchenQual', 'Functional', 'SaleType']
cat_columns[mode_cols] = cat_columns[mode_cols].fillna(cat_columns.mode().iloc[0])
null_cat_columns = cat_columns.isnull().sum()
null_cat_columns
null_num_columns = num_columns.isnull().sum()
null_num_columns
print(num_columns['GarageYrBlt'].median())
print(num_columns['LotFrontage'].median())
num_columns['GarageYrBlt'] = num_columns['GarageYrBlt'].fillna(1979)
num_columns['LotFrontage'] = num_columns['LotFrontage'].fillna(68)
num_columns = num_columns.fillna(0)
num_columns.isnull().sum()
num_columns['House_Age'] = num_columns['YrSold'] - num_columns['YearBuilt']
num_columns['House_Age'].describe()
neg_house = num_columns[num_columns['House_Age'] < 0]
neg_house
neg_house['YrSold']
num_columns.loc[num_columns['YrSold'] < num_columns['YearRemodAdd'], 'YrSold'] = 2009
num_columns['House_Age'] = num_columns['YrSold'] - num_columns['YearBuilt']
num_columns['House_Age'].describe()
num_columns['TotalBsmtBath'] = num_columns['BsmtFullBath'] * 0.5 + num_columns['BsmtHalfBath']
num_columns['TotalBath'] = num_columns['FullBath'] * 0.5 + num_columns['HalfBath']
num_columns['TotalSA'] = num_columns['TotalBsmtSF'] + num_columns['1stFlrSF'] + num_columns['2ndFlrSF']
num_columns.head()
cat_columns.head()
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
for features in cat_columns.columns:
    cat_columns[features] = encoder.fit_transform(cat_columns[features])
cat_columns.head()
df_final = pd.concat([cat_columns, num_columns], axis=1, sort=False)
df_final.head()
df_final = df_final.drop(['Id'], axis=1)
df_train = df_final[df_final['train'] == 1]
df_train = df_train.drop(['train'], axis=1)
df_test = df_final[df_final['train'] == 0]
df_test = df_test.drop(['SalePrice'], axis=1)
df_test = df_test.drop(['train'], axis=1)
df_train.head()
df_test.head()
target = df_train['SalePrice']
df_train = df_train.drop(['SalePrice'], axis=1)
(x_train, x_test, y_train, y_test) = train_test_split(df_train, target, test_size=0.33, random_state=0)
xgb = XGBRegressor(booster='gbtree', colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.6, gamma=0, importance_type='gain', learning_rate=0.01, max_delta_step=0, max_depth=4, min_child_weight=1.5, n_estimators=2400, n_jobs=1, nthread=None, objective='reg:linear', reg_alpha=0.6, reg_lambda=0.6, scale_pos_weight=1, silent=None, subsample=0.8, verbosity=1)
lgbm = LGBMRegressor(objective='regression', num_leaves=4, learning_rate=0.01, n_estimators=12000, max_bin=200, bagging_fraction=0.75, bagging_freq=5, bagging_seed=7, feature_fraction=0.4)