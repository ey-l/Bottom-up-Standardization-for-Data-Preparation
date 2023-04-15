import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn import metrics
iowa_file_path = '_data/input/house-prices-advanced-regression-techniques/train.csv'
home_data = pd.read_csv(iowa_file_path)
print(home_data)
home_data.describe()
home_data.isnull().sum()
y = home_data.SalePrice
print(y)
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]
print(X)
X.shape
(X_train, X_test, Y_train, Y_test) = train_test_split(X, y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)