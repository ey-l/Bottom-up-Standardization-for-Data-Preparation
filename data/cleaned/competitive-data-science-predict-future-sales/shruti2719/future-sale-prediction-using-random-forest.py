import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sklearn
from sklearn import *
import nltk, datetime
from sklearn import ensemble, metrics, preprocessing
itemcat = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
result = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
itemcat.info()
items.info()
train.info()
shops.info()
test.info()
result.info()
train.head()
train.date_block_num.max()
test.head()
train.describe()
print('train', train.shape)
print('test', test.shape)
train['date'] = pd.to_datetime(train['date'], format='%d.%m.%Y')
train['month'] = train['date'].dt.month
train['year'] = train['date'].dt.year
train = train.drop(['date', 'item_price'], axis=1)
train = train.groupby([c for c in train.columns if c not in ['item_cnt_day']], as_index=False)[['item_cnt_day']].sum()
train = train.rename(columns={'item_cnt_day': 'item_cnt_month'})
train.head()
shop_item_mean = train[['shop_id', 'item_id', 'item_cnt_month']].groupby(['shop_id', 'item_id'], as_index=False)[['item_cnt_month']].mean()
shop_item_mean = shop_item_mean.rename(columns={'item_cnt_month': 'item_cnt_month_mean'})
train = pd.merge(train, shop_item_mean, how='left', on=['shop_id', 'item_id'])
train.head()
shop_prev_month = train[train['date_block_num'] == 33][['shop_id', 'item_id', 'item_cnt_month']]
shop_prev_month = shop_prev_month.rename(columns={'item_cnt_month': 'item_cnt_prev_month'})
shop_prev_month.head()
train = pd.merge(train, shop_prev_month, how='left', on=['shop_id', 'item_id']).fillna(0.0)
train = pd.merge(train, items, how='left', on='item_id')
train = pd.merge(train, itemcat, how='left', on='item_category_id')
train = pd.merge(train, shops, how='left', on='shop_id')
train.head()
test['month'] = 11
test['year'] = 2015
test['date_block_num'] = 34
test = pd.merge(test, shop_item_mean, how='left', on=['shop_id', 'item_id']).fillna(0.0)
test = pd.merge(test, shop_prev_month, how='left', on=['shop_id', 'item_id']).fillna(0.0)
test = pd.merge(test, items, how='left', on=['item_id'])
test = pd.merge(test, itemcat, how='left', on=['item_category_id'])
test = pd.merge(test, shops, how='left', on='shop_id')
test['item_cnt_month'] = 0
test.head()
for c in ['shop_name', 'item_name', 'item_category_name']:
    lbl = preprocessing.LabelEncoder()