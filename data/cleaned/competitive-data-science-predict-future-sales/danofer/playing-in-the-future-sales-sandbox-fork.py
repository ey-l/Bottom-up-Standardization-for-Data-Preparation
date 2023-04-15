import numpy as np
import pandas as pd
from sklearn import *
import nltk, datetime
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
submission = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
item_cats = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
print('train:', train.shape, 'test:', test.shape)
train = train.loc[train.item_cnt_day <= 1001]
train = train.loc[train.item_price < 100001]
subset = ['date', 'shop_id', 'item_id', 'item_cnt_day']
print(train.duplicated(subset=subset).value_counts())
train.drop_duplicates(subset=subset, inplace=True)
print('Orig train shape:', train.shape[0])
test_shops = test.shop_id.unique()
test_items = test.item_id.unique()
print(len(test_shops), 'Test shops')
print(len(test_items), 'Test items')
train = train[train.shop_id.isin(test_shops)]
train = train[train.item_id.isin(test_items)]
print('New train (intersecting test) shape:', train.shape)
train = train.loc[train.item_cnt_day != -1]
[c for c in train.columns if c not in test.columns]
train.head()
item_grp = item_cats['item_category_name'].apply(lambda x: str(x).split(' ')[0])
item_cats['item_group'] = item_grp
items = pd.merge(items, item_cats.loc[:, ['item_category_id', 'item_group']], on=['item_category_id'], how='left')
shops['city'] = shops.shop_name.apply(lambda x: str.replace(x, '!', '')).apply(lambda x: x.split(' ')[0])
train['date'] = pd.to_datetime(train['date'], format='%d.%m.%Y')
train['month'] = train['date'].dt.month
train.drop(['item_price'], axis=1, inplace=True)
train = train.groupby([c for c in train.columns if c not in ['item_cnt_day']], as_index=False)[['item_cnt_day']].sum()
train = train.rename(columns={'item_cnt_day': 'item_cnt_month'})
shop_item_monthly_mean = train[['shop_id', 'item_id', 'item_cnt_month']].groupby(['shop_id', 'item_id'], as_index=False)[['item_cnt_month']].mean()
shop_item_monthly_mean = shop_item_monthly_mean.rename(columns={'item_cnt_month': 'item_cnt_month_mean'})
train = pd.merge(train, shop_item_monthly_mean, how='left', on=['shop_id', 'item_id'])
shop_item_prev_month = train[train['date_block_num'] == 33][['shop_id', 'item_id', 'item_cnt_month']]
shop_item_prev_month = shop_item_prev_month.rename(columns={'item_cnt_month': 'item_cnt_prev_month'})
shop_item_prev_month.head()
train = pd.merge(train, shop_item_prev_month, how='left', on=['shop_id', 'item_id'])
train = pd.merge(train, items, how='left', on='item_id')
train = pd.merge(train, item_cats, how='left', on='item_category_id')
train = pd.merge(train, shops, how='left', on='shop_id')
train.head()
print(train.shape[0])
train.drop_duplicates(inplace=True)
print(train.shape[0])
test['month'] = 11
test['date_block_num'] = 34
test = pd.merge(test, shop_item_monthly_mean, how='left', on=['shop_id', 'item_id'])
test = pd.merge(test, shop_item_prev_month, how='left', on=['shop_id', 'item_id'])
test = pd.merge(test, items, how='left', on='item_id')
test = pd.merge(test, item_cats, how='left', on='item_category_id')
test = pd.merge(test, shops, how='left', on='shop_id')
test.head()
print(test.shape[0])
test.drop_duplicates(subset=['ID'], inplace=True)
print(test.shape[0])
train['shop_item'] = train['shop_id'].astype(str) + '-' + train['item_id'].astype(str)
test['shop_item'] = test['shop_id'].astype(str) + '-' + test['item_id'].astype(str)
test['date'] = datetime.date(2015, 11, 30)
train.shape
test.shape
set(train.columns) - set(test.columns)
set(test.columns) - set(train.columns)


df_all = pd.concat((train, test), axis=0, ignore_index=True)
stores_hm = df_all.pivot_table(index='shop_id', columns='item_category_id', values='item_cnt_month', aggfunc='count', fill_value=0)
(fig, ax) = plt.subplots(figsize=(10, 10))
_ = sns.heatmap(stores_hm, ax=ax, cbar=False)