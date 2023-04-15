import pandas as pd
main_file_path = '_data/input/house-prices-advanced-regression-techniques/train.csv'
data = pd.read_csv(main_file_path)
data.describe()
y = data.SalePrice
y.head()
predictors = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = data[predictors]
X.head()
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
(train_X, val_X, train_y, val_y) = train_test_split(X, y, random_state=0)
my_model = DecisionTreeRegressor()