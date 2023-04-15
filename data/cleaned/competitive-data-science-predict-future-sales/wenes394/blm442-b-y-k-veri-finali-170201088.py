import numpy as np
import pandas as pd
from sklearn import *
import nltk, datetime
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
submission = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
item_cats = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
train.head()
train.shape
test.shape
train['date'] = pd.to_datetime(train['date'], format='%d.%m.%Y')
train['month'] = train['date'].dt.month
train['year'] = train['date'].dt.year
train = train.drop(['date', 'item_price'], axis=1)
train = train.groupby([column for column in train.columns if column not in ['item_cnt_day']], as_index=False)[['item_cnt_day']].sum()
train = train.rename(columns={'item_cnt_day': 'item_count_monthly'})
train.head(10)
shop_item_ort = train[['shop_id', 'item_id', 'item_count_monthly']].groupby(['shop_id', 'item_id'], as_index=False)[['item_count_monthly']].mean()
shop_item_ort = shop_item_ort.rename(columns={'item_count_monthly': 'item_count_monthly_mean'})
train = pd.merge(train, shop_item_ort, how='left', on=['shop_id', 'item_id'])
train.head(9)
shop_son_ay = train[train['date_block_num'] == 33][['shop_id', 'item_id', 'item_count_monthly']]
shop_son_ay = shop_son_ay.rename(columns={'item_count_monthly': 'item_count_son_ay'})
shop_son_ay.head()
train = pd.merge(train, shop_son_ay, how='left', on=['shop_id', 'item_id']).fillna(0.0)
train = pd.merge(train, items, how='left', on='item_id')
train = pd.merge(train, item_cats, how='left', on='item_category_id')
train = pd.merge(train, shops, how='left', on='shop_id')
train.head()
test['month'] = 11
test['year'] = 2015
test['date_block_num'] = 34
test = pd.merge(test, shop_item_ort, how='left', on=['shop_id', 'item_id']).fillna(0.0)
test = pd.merge(test, shop_son_ay, how='left', on=['shop_id', 'item_id']).fillna(0.0)
test = pd.merge(test, items, how='left', on='item_id')
test = pd.merge(test, item_cats, how='left', on='item_category_id')
test = pd.merge(test, shops, how='left', on='shop_id')
test['item_count_monthly'] = 0.0
test.head()
for column in ['shop_name', 'item_name', 'item_category_name']:
    label = preprocessing.LabelEncoder()