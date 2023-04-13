import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import sklearn.metrics as metrics
import math
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
c_test = _input0.copy()
c_train = _input1.copy()
_input1.head()
_input0.head()
c_train['train'] = 1
c_test['train'] = 0
df = pd.concat([c_train, c_test], axis=0, sort=False)
df.head()
df_col_name = df.columns
values = []
for col in df_col_name:
    values.append(df[col].isnull().sum() / df.shape[0] * 100)
df_nan = pd.DataFrame(df_col_name, columns=['Feature'])
df_nan['percent'] = values
df_nan
df_nan = df_nan[df_nan.percent > 50]
df_nan.sort_values('percent', ascending=False)
df = df.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)
df_object = df.select_dtypes(include=['object'])
df_numerical = df.select_dtypes(exclude=['object'])
null_counts = df_object.isnull().sum()
null_counts
none_col = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'GarageType', 'GarageFinish', 'GarageQual', 'FireplaceQu', 'GarageCond']
df_object[none_col] = df_object[none_col].fillna('None')
df_object = df_object.fillna('mode()')
null_counts = df_numerical.isnull().sum()
null_counts
df_numerical['LotFrontage'] = df_numerical['LotFrontage'].fillna(method='ffill')
df_numerical['GarageYrBlt'] = df_numerical['GarageYrBlt'].fillna(df_numerical['YrSold'], inplace=False)
df_numerical = df_numerical.fillna(0, inplace=False)
df_numerical['Age_House'] = df_numerical['YrSold'] - df_numerical['YearBuilt']
df_numerical[df_numerical['Age_House'] < 0]
df_numerical.loc[df_numerical['YrSold'] < df_numerical['YearBuilt'], 'YrSold'] = 2009
df_numerical['Age_House'] = df_numerical['YrSold'] - df_numerical['YearBuilt']
from sklearn.preprocessing import StandardScaler, LabelEncoder
le = LabelEncoder()
for col in df_object.columns.tolist():
    df_object[col] = le.fit_transform(df_object[col])
df_final = pd.concat([df_object, df_numerical], axis=1, sort=False)
df_final.head()
df_final = df_final.drop(['Id'], axis=1)
_input1 = df_final[df_final['train'] == 1]
_input1 = _input1.drop(['train'], axis=1)
_input0 = df_final[df_final['train'] == 0]
_input0 = _input0.drop(['SalePrice'], axis=1)
_input0 = _input0.drop(['train'], axis=1)
X = _input1.drop(['SalePrice'], axis=1)
y = _input1['SalePrice']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=1)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()