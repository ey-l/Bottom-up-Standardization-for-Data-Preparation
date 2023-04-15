import pandas as pd
train_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
features = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'BsmtFinSF1', '2ndFlrSF', 'GarageArea', '1stFlrSF', 'YearBuilt']
X_train = train_data[features]
y_train = train_data['SalePrice']
final_X_test = test_data[features]
X_train = X_train.fillna(X_train.mean())
final_X_test = final_X_test.fillna(final_X_test.mean())
from catboost import CatBoostRegressor
regressor = CatBoostRegressor(loss_function='RMSE', verbose=False)