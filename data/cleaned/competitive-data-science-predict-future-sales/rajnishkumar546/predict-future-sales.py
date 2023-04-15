import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import datetime
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
train.head(5)
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
submission = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
item_cats = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
train.shape
train.describe()
test.shape
train['date'] = pd.to_datetime(train['date'], format='%d.%m.%Y')
train['month'] = train['date'].dt.month
train['year'] = train['date'].dt.year
train = train.drop(['date', 'item_price'], axis=1)
train.head()
train = train.groupby([c for c in train.columns if c not in ['item_cnt_day']], as_index=False)[['item_cnt_day']].sum()
train = train.rename(columns={'item_cnt_day': 'item_cnt_month'})
train.head(5)
shop_item_mean = train[['shop_id', 'item_id', 'item_cnt_month']].groupby(['shop_id', 'item_id'], as_index=False)[['item_cnt_month']].mean()
shop_item_mean = shop_item_mean.rename(columns={'item_cnt_month': 'item_cnt_month_mean'})
shop_item_mean
train = pd.merge(train, shop_item_mean, how='left', on=['shop_id', 'item_id'])
train.head()
train.shape
shop_prev_month = train[train['date_block_num'] == 33][['shop_id', 'item_id', 'item_cnt_month']]
shop_prev_month = shop_prev_month.rename(columns={'item_cnt_month': 'item_cnt_prev_month'})
shop_prev_month.head()
train = pd.merge(train, shop_prev_month, how='left', on=['shop_id', 'item_id']).fillna(0.0)
train = pd.merge(train, items, how='left', on='item_id')
train = pd.merge(train, item_cats, how='left', on=['item_category_id'])
train = pd.merge(train, shops, how='left', on=['shop_id'])
train.head()
test['month'] = 11
test['year'] = 2015
test['date_block_num'] = 34
test = pd.merge(test, shop_item_mean, how='left', on=['shop_id', 'item_id']).fillna(0.0)
test = pd.merge(test, shop_prev_month, how='left', on=['shop_id', 'item_id']).fillna(0.0)
test = pd.merge(test, items, how='left', on='item_id')
test = pd.merge(test, item_cats, how='left', on='item_category_id')
test = pd.merge(test, shops, how='left', on='shop_id')
test.head()
test['item_cnt_month'] = 0
from sklearn.preprocessing import LabelEncoder
for c in ['shop_name', 'item_name', 'item_category_name']:
    lbl = LabelEncoder()