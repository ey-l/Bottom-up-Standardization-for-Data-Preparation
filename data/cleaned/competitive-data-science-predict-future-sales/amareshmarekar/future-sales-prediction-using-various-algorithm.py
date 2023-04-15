import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import sklearn

from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from scipy import stats
import sklearn.metrics as sm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import catboost as ctb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor

import matplotlib.pyplot as plt
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
items.head()
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
test.head()
categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
categories.head(20)
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
shops.head()
sale_train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
sale_train.head()
print('----------Top-5- Record----------')
print(sale_train.head(5))
print('-----------Information-----------')
print(sale_train.info())
print('-----------Data Types-----------')
print(sale_train.dtypes)
print('----------Missing value-----------')
print(sale_train.isnull().sum())
print('----------Null value-----------')
print(sale_train.isna().sum())
print('----------Shape of Data----------')
print(sale_train.shape)
print('Number of duplicates:', len(sale_train[sale_train.duplicated()]))
sale_train.info()
sale_train.describe()
sale_train.hist(figsize=(8, 8), bins=6)

import seaborn as sns
sns.displot(sale_train['item_cnt_day'])
sale_train['item_cnt_day'].describe()
sale_train = sale_train.drop(columns=['date', 'date_block_num', 'item_price'])
sale_train.head()
y = sale_train['item_cnt_day']
x = sale_train.drop(columns=['item_cnt_day'])
x.head()
(X_train, X_test, Y_train, Y_test) = train_test_split(x, y, test_size=0.3, random_state=60, shuffle=True)
print(len(X_train))
print(len(X_test))
linear_model = LinearRegression()