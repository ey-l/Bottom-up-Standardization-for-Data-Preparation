import pandas as pd
import numpy as np
import xgboost as xgb
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col=0)
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col=0)
X_train = _input1.select_dtypes(include=['number']).copy()
X_train = X_train.drop(['SalePrice'], axis=1)
y_train = _input1['SalePrice']
X_test = _input0.select_dtypes(include=['number']).copy()
for df in (X_train, X_test):
    df['n_bathrooms'] = df['BsmtFullBath'] + df['BsmtHalfBath'] * 0.5 + df['FullBath'] + df['HalfBath'] * 0.5
    df['area_with_basement'] = df['GrLivArea'] + df['TotalBsmtSF']
regressor = xgb.XGBRegressor(eval_metric='rmsle')
from sklearn.model_selection import GridSearchCV
param_grid = {'max_depth': [4, 5], 'n_estimators': [500, 600, 700], 'learning_rate': [0.01, 0.015]}