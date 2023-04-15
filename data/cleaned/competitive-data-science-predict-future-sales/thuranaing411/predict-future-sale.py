import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from scipy import optimize, stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error
X_train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
X_test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
X_train.head()
X_train.drop(['date_block_num', 'item_price'], axis=1, inplace=True)
X_train['date'] = pd.to_datetime(X_train['date'], dayfirst=True)
X_train['date'] = X_train['date'].apply(lambda x: x.strftime('%Y-%m'))
X_train.head()
X = X_train.groupby(['date', 'shop_id', 'item_id']).sum()
X = X.pivot_table(index=['shop_id', 'item_id'], columns='date', values='item_cnt_day', fill_value=0)
X.reset_index(inplace=True)
X.head()
X_test = pd.merge(X_test, X, on=['shop_id', 'item_id'], how='left')
X_test.drop(['2013-01'], axis=1, inplace=True)
X_test = X_test.fillna(0)
X_test.head()
Y = X['2015-10'].values
X_T = X.drop(['2015-10'], axis=1)
X_test = X_test
(x_train, x_test, y_train, y_test) = train_test_split(X_T, Y, test_size=0.2, random_state=1)
lin_reg = LinearRegression(normalize=True)