import pandas as pd
from sklearn.model_selection import train_test_split
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
y = _input1.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
x = _input1[features].copy()
x_test = _input0[features].copy()
(X_train, X_valid, y_train, y_valid) = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=0)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
model = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)