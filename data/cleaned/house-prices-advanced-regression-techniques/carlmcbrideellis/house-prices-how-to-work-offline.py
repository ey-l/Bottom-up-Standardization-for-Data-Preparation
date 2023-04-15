import pandas as pd
import numpy as np
train_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF']
X_train = train_data[features]
y_train = train_data['SalePrice']
final_X_test = test_data[features]
X_train = X_train.fillna(X_train.mean())
final_X_test = final_X_test.fillna(final_X_test.mean())
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, max_depth=7)