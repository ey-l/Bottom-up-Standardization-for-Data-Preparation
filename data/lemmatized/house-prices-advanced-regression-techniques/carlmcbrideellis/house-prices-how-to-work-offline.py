import pandas as pd
import numpy as np
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF']
X_train = _input1[features]
y_train = _input1['SalePrice']
final_X_test = _input0[features]
X_train = X_train.fillna(X_train.mean())
final_X_test = final_X_test.fillna(final_X_test.mean())
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, max_depth=7)