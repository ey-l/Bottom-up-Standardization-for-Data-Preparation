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
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from keras.utils import np_utils
df_train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
df_shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
df_items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
df_item_categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
df_test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
df_train.head()
df_test.head()
df_train.info()
df_train.describe()
df_train.isnull().sum()
df_test.isna().sum()
import pandas_profiling as pp
pp.ProfileReport(df_train)
import pandas_profiling as pp
pp.ProfileReport(df_shops)
import pandas_profiling as pp
pp.ProfileReport(df_items)
import pandas_profiling as pp
pp.ProfileReport(df_test)
df_train.drop(['date_block_num', 'item_price'], axis=1, inplace=True)
df_train.head()
df_train['date'] = pd.to_datetime(df_train['date'], dayfirst=True)
df_train['date'] = df_train['date'].apply(lambda x: x.strftime('%Y-%m'))
df_train.head()
df = df_train.groupby(['date', 'shop_id', 'item_id']).sum()
df = df.pivot_table(index=['shop_id', 'item_id'], columns='date', values='item_cnt_day', fill_value=0)
df.reset_index(inplace=True)
df.head().T
df_test = pd.merge(df_test, df, on=['shop_id', 'item_id'], how='left')
df_test.drop(['ID', '2013-01'], axis=1, inplace=True)
df_test = df_test.fillna(0)
df_test.head().T
Y_train = df['2015-10'].values
X_train = df.drop(['2015-10'], axis=1)
X_test = df_test
print(X_train.shape, Y_train.shape)
print(X_test.shape)
(x_train, x_test, y_train, y_test) = train_test_split(X_train, Y_train, test_size=0.2, random_state=1)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
y_test

LR = LinearRegression()