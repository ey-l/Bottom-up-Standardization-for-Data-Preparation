import pandas as pd
import numpy as np
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col=0)
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col=0)
target = 'SalePrice'
X_train = _input1.select_dtypes(include=['number']).copy()
X_train = X_train.drop([target], axis=1)
y_train = _input1[target]
X_test = _input0.select_dtypes(include=['number']).copy()
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, max_depth=10)
from sklearn.feature_selection import RFE
n_features_to_select = 1
rfe = RFE(regressor, n_features_to_select=n_features_to_select)