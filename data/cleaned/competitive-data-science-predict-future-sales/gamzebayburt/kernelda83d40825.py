import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import matplotlib.pyplot as plt
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
item_categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
submission = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
train.head()
print('Train data shape: ', train.shape)
print('Test data shape: ', test.shape)
train = train[train['item_cnt_day'] >= 0]
train = train[train['item_cnt_day'] < 20]
print(train.item_cnt_day.describe())
train['date'] = pd.to_datetime(train['date'], format='%d.%m.%Y')
train['month'] = train['date'].dt.month
train['year'] = train['date'].dt.year
train = train.drop(['date', 'item_price'], axis=1)
train.head()
train = train.groupby(['date_block_num', 'shop_id', 'item_id', 'month', 'year'], as_index=False)[['item_cnt_day']].sum()
train.columns = ['date_block_num', 'shop_id', 'item_id', 'month', 'year', 'item_cnt_month']
train.head()
shop_item_mean = train[['item_id', 'shop_id', 'item_cnt_month']].groupby(['item_id', 'shop_id'], as_index=False)[['item_cnt_month']].mean()
shop_item_mean.columns = ['item_id', 'shop_id', 'item_cnt_month_mean']
shop_item_mean.head()
train = pd.merge(train, shop_item_mean, how='left', on=['shop_id', 'item_id'])
train.head()
last_month_sales = train[train['date_block_num'] == 33][['shop_id', 'item_id', 'item_cnt_month']]
last_month_sales.columns = ['shop_id', 'item_id', 'item_cnt_last_month']
last_month_sales.head()
train = pd.merge(train, last_month_sales, how='left', on=['shop_id', 'item_id']).fillna(0.0)
train.head()
train = pd.merge(train, items, how='left', on=['item_id'])
train.head()
train = train.drop('item_name', axis=1)
train.head()
test['date_block_num'] = 34
test['year'] = 2015
test['month'] = 11
test = pd.merge(test, shop_item_mean, how='left', on=['shop_id', 'item_id']).fillna(0.0)
test = pd.merge(test, last_month_sales, how='left', on=['shop_id', 'item_id']).fillna(0.0)
test = pd.merge(test, items, how='left', on=['item_id'])
test = test.drop('item_name', axis=1)
test['item_cnt_month'] = 0.0
test.head()
train.corr()
feature_cols = [c for c in train.columns if c not in ['item_cnt_month']]
X_train = train[train['date_block_num'] < 33]
y_train = X_train['item_cnt_month']
X_test = train[train['date_block_num'] == 33]
y_test = X_test['item_cnt_month']
X_train = X_train[feature_cols]
X_test = X_test[feature_cols]
regressor = ensemble.ExtraTreesRegressor(n_estimators=30, max_depth=15, n_jobs=-1, random_state=18)