import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from scipy import optimize, stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error
_input0 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
_input2 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
_input0.head()
_input0 = _input0.drop(['date_block_num', 'item_price'], axis=1, inplace=False)
_input0['date'] = pd.to_datetime(_input0['date'], dayfirst=True)
_input0['date'] = _input0['date'].apply(lambda x: x.strftime('%Y-%m'))
_input0.head()
X = _input0.groupby(['date', 'shop_id', 'item_id']).sum()
X = X.pivot_table(index=['shop_id', 'item_id'], columns='date', values='item_cnt_day', fill_value=0)
X = X.reset_index(inplace=False)
X.head()
_input2 = pd.merge(_input2, X, on=['shop_id', 'item_id'], how='left')
_input2 = _input2.drop(['2013-01'], axis=1, inplace=False)
_input2 = _input2.fillna(0)
_input2.head()
Y = X['2015-10'].values
X_T = X.drop(['2015-10'], axis=1)
_input2 = _input2
(x_train, x_test, y_train, y_test) = train_test_split(X_T, Y, test_size=0.2, random_state=1)
lin_reg = LinearRegression(normalize=True)