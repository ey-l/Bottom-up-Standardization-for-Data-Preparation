import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
iowa_file_path = '../input/train.csv'
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1.describe()
_input1.head()
y = _input1.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = _input1[features]
X.head()
(train_X, val_X, train_y, val_y) = train_test_split(X, y, random_state=1)
iowa_model = DecisionTreeRegressor(random_state=1)