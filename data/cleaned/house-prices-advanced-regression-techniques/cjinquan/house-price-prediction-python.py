import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
iowa_file_path = '_data/input/house-prices-advanced-regression-techniques/train.csv'
home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice
features = ['MSSubClass', 'LotArea', 'HalfBath', 'ScreenPorch', 'OverallQual', 'YearBuilt', 'OverallCond', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'FullBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'WoodDeckSF', 'OpenPorchSF', '3SsnPorch', 'PoolArea', 'MoSold', 'YrSold']
X = home_data[features]
X.head()
(train_X, val_X, train_y, val_y) = train_test_split(X, y, random_state=1)
rf_model = RandomForestRegressor(random_state=1)