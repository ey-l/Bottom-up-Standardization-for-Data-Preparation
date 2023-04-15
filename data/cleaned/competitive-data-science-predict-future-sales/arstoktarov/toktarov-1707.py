import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import os
import sys
import re
import gc
import time
import datetime
import warnings
from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import plot_importance
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
item_categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
sales_train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
ss = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
items.head()
items['item_name'].value_counts()
items['item_category_id'].value_counts()
item_with_categories = pd.merge(items, item_categories, on='item_category_id')
item_with_categories.head()
sales_train.head()
shops.head()
shops_with_sales = pd.merge(shops, sales_train, on='shop_id')
shops_with_sales.head()
train = pd.merge(shops_with_sales, item_with_categories, on='item_id')
train.head()
train['sales'] = train['item_price'] * train['item_cnt_day']
train.head()
train
train['item_id'].value_counts()
train['date'] = pd.to_datetime(train.date, format='%d.%m.%Y')
plt.scatter(train.index, train.date_block_num)
train.groupby('date_block_num').agg({'item_cnt_day': 'sum'}).plot(figsize=(15, 6), title='Items transacted per day')
train['item_cnt_day'].plot(figsize=(10, 6))
index_cols = ['shop_id', 'item_id', 'date_block_num']
grid = []
for block_num in train['date_block_num'].unique():
    cur_shops = train.loc[train['date_block_num'] == block_num, 'shop_id'].unique()
    cur_items = train.loc[train['date_block_num'] == block_num, 'item_id'].unique()
    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])), dtype='int32'))
grid = pd.DataFrame(np.vstack(grid), columns=index_cols, dtype=np.int32)
grid.head()
train_m = train.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_day': 'sum', 'item_price': np.mean}).reset_index()
train_m = pd.merge(grid, train_m, on=['date_block_num', 'shop_id', 'item_id'], how='left').fillna(0)
train_m = pd.merge(train_m, items, on='item_id', how='left')
for type_id in ['item_id', 'shop_id', 'item_category_id']:
    for (column_id, aggregator, aggtype) in [('item_price', np.mean, 'avg'), ('item_cnt_day', np.sum, 'sum'), ('item_cnt_day', np.mean, 'avg')]:
        mean_df = train.groupby([type_id, 'date_block_num']).aggregate(aggregator).reset_index()[[column_id, type_id, 'date_block_num']]
        mean_df.columns = [type_id + '_' + aggtype + '_' + column_id, type_id, 'date_block_num']
        train_m = pd.merge(train_m, mean_df, on=['date_block_num', type_id], how='left')
        del mean_df
        gc.collect()
del train
gc.collect()
for f in train_m.columns:
    if 'item_cnt' in f:
        train_m[f] = train_m[f].fillna(0)
    elif 'item_price' in f:
        train_m[f] = train_m[f].fillna(train_m[f].median())
train_m.info(verbose=False)
max_clip = 30
train_m['item_cnt_day'] = train_m['item_cnt_day'].clip(0, max_clip).astype(np.float16)
train_m['month'] = train_m['date_block_num'] % 12
train_m['month'] = train_m['month'].astype(np.int8)
days = pd.Series([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
train_m['days'] = train_m['month'].map(days).astype(np.int8)
test['month'] = 11
test['month'] = test['month'].astype(np.int8)
test['days'] = 30
test['days'] = test['days'].astype(np.int8)
train_m = train_m[train_m['date_block_num'] > 12]
train_set = train_m[train_m['date_block_num'] < 33]
val_set = train_m[train_m['date_block_num'] == 33]
print(train_set.shape)
print(val_set.shape)
print(test.shape)
train_x = train_set.drop(['item_cnt_day'], axis=1)
train_y = train_set['item_cnt_day']
val_x = val_set.drop(['item_cnt_day'], axis=1)
val_y = val_set['item_cnt_day']
features = list(train_x.columns.values)
train_x.drop(columns=['item_name'], inplace=True)
gc.collect()
lm = linear_model.Ridge()