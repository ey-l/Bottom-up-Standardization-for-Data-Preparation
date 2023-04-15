import pandas as pd
import numpy as np
train_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col=0)
test_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col=0)
target = 'SalePrice'
X_train = train_data.select_dtypes(include=['number']).copy()
X_train = X_train.drop([target], axis=1)
y_train = train_data[target]
X_test = test_data.select_dtypes(include=['number']).copy()
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, max_depth=10)
from sklearn.feature_selection import RFE
n_features_to_select = 1
rfe = RFE(regressor, n_features_to_select=n_features_to_select)