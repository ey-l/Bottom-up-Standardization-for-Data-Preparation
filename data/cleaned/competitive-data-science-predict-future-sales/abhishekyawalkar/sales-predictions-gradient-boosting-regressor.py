import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import keras
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
items.head()
item_categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
item_categories.head()
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
shops.head()
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
train.head()
train.shape
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
Id = test['ID']
test.head()
test.shape
train.info()
train['date'] = pd.to_datetime(train['date'])
train.isna().sum()
train['date'] = train['date'].apply(lambda x: x.strftime('%Y-%m'))
train.head().sort_values(by='date')
train.drop(['date_block_num', 'item_price'], axis=1, inplace=True)
train.head().sort_values(by='date')
df = train.groupby(['date', 'shop_id', 'item_id']).sum()
df
df = train.pivot_table(index=['shop_id', 'item_id'], columns='date', values='item_cnt_day', fill_value=0)
df.reset_index(inplace=True)
df.head()
test_df = pd.merge(test, df, on=['shop_id', 'item_id'], how='left')
test_df.drop(['ID', '2013-01'], axis=1, inplace=True)
test_df = test_df.fillna(0)
test_df.head()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
Y = df['2015-10'].values
X = df.drop(['2015-10'], axis=1)
test_full = test_df
(X_train, X_test, y_train, y_test) = train_test_split(X, Y, test_size=0.2, random_state=42)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
lr = LinearRegression()