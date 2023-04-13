import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, r2_score
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (12, 8)
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.shape
_input0.shape
_input1.head()
_input1['TotalSF'] = _input1['TotalBsmtSF'] + _input1['1stFlrSF'] + _input1['2ndFlrSF']
for column in ('TotalBsmtSF', '1stFlrSF', '2ndFlrSF'):
    _input1[column] = _input1[column] / _input1['TotalSF']
_input1['BsmtUnfSF'] = _input1['BsmtUnfSF'] / _input1['TotalBsmtSF']
_input1['LowQualFinSF'] = _input1['LowQualFinSF'] / _input1['TotalSF']
y_train = _input1['SalePrice'] / _input1['TotalSF']
x_train = _input1.drop(['Id', 'SalePrice'], axis='columns')
missing_vals = _input1.isna().sum()
print(missing_vals[missing_vals > 0])
columns_to_drop = missing_vals[missing_vals > 100].index
x_train = x_train.drop(columns_to_drop, axis='columns')
print(columns_to_drop)
cat_columns = list(x_train.select_dtypes(include=['category', 'object']).columns)
num_columns = list(x_train.select_dtypes(include=['number']).columns)
imp_median = SimpleImputer(strategy='median')
imp_freq = SimpleImputer(strategy='most_frequent')
ohe = OneHotEncoder(sparse=False)
cat_transformer = make_pipeline(imp_freq, ohe)
col_transformer = make_column_transformer((imp_median, num_columns), (cat_transformer, cat_columns), remainder='passthrough')
regr_model = RandomForestRegressor()
pipe_RF = make_pipeline(col_transformer, regr_model)