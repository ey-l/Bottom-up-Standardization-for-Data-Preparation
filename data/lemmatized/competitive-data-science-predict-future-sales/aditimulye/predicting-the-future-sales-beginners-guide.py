import pandas as pd
import numpy as np
from scipy import optimize, stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
_input0 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
_input1 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
_input4 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
_input3 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
_input2 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
_input0.head()
_input0.dtypes
_input0.describe()
_input0.isnull().sum()
_input0 = _input0.drop(['date_block_num', 'item_price'], axis=1, inplace=False)
_input0['date'] = pd.to_datetime(_input0['date'], dayfirst=True)
_input0['date'] = _input0['date'].apply(lambda x: x.strftime('%Y-%m'))
_input0.head()
df = _input0.groupby(['date', 'shop_id', 'item_id']).sum()
df = df.pivot_table(index=['shop_id', 'item_id'], columns='date', values='item_cnt_day', fill_value=0)
df = df.reset_index(inplace=False)
df.head()
_input2 = pd.merge(_input2, df, on=['shop_id', 'item_id'], how='left')
_input2 = _input2.drop(['ID', '2013-01'], axis=1, inplace=False)
_input2 = _input2.fillna(0)
_input2.head()
Y_train = df['2015-10'].values
X_train = df.drop(['2015-10'], axis=1)
X_test = _input2
print(X_train.shape, Y_train.shape)
print(X_test.shape)
(x_train, x_test, y_train, y_test) = train_test_split(X_train, Y_train, test_size=0.2, random_state=101)
print('Train set:', x_train.shape, y_train.shape)
print('Test set:', x_test.shape, y_test.shape)
LR = LinearRegression()