import pandas as pd
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
features = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'BsmtFinSF1', '2ndFlrSF', 'GarageArea', '1stFlrSF', 'YearBuilt']
X_train = _input1[features]
y_train = _input1['SalePrice']
final_X_test = _input0[features]
X_train = X_train.fillna(X_train.mean())
final_X_test = final_X_test.fillna(final_X_test.mean())
from catboost import CatBoostRegressor
regressor = CatBoostRegressor(loss_function='RMSE', verbose=False)