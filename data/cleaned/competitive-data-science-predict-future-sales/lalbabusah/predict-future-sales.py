import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
from scipy import optimize, stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Flatten, Dropout
from tensorflow.keras.layers import LeakyReLU, PReLU, ELU
from keras.utils import np_utils
df_train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
print(df_train.shape)
df_train.head()
df_shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
print(df_shops.shape)
df_shops.head()
df_items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
print(df_items.shape)
df_items.head()
df_item_categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
print(df_item_categories.shape)
df_item_categories.head()
df_test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
print(df_test.shape)
df_test.head()
df_train.head()
df_train.dtypes
df_train.isnull().sum()
df_train.drop(['date_block_num', 'item_price'], axis=1, inplace=True)
df_train['date'] = pd.to_datetime(df_train['date'], dayfirst=True)
df_train['date'] = df_train['date'].apply(lambda x: x.strftime('%Y-%m'))
df_train.head()
df = df_train.groupby(['date', 'shop_id', 'item_id']).sum()
df = df.pivot_table(index=['shop_id', 'item_id'], columns='date', values='item_cnt_day', fill_value=0)
df.reset_index(inplace=True)
df.head()
df_test = pd.merge(df_test, df, on=['shop_id', 'item_id'], how='left')
df_test.drop(['ID', '2013-01'], axis=1, inplace=True)
df_test = df_test.fillna(0)
df_test.head()
Y_train = df['2015-10'].values
X_train = df.drop(['2015-10'], axis=1)
X_test = df_test
print(X_train.shape, Y_train.shape)
print(X_test.shape)