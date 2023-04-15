import numpy as np
import pandas as pd
from sklearn import *
import nltk, datetime
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
submission = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
item_cats = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
print('train:', train.shape, 'test:', test.shape)
[c for c in train.columns if c not in test.columns]
train.head()
test.head()
feature_cnt = 25
tfidf = feature_extraction.text.TfidfVectorizer(max_features=feature_cnt)
items['item_name_len'] = items['item_name'].map(len)
items['item_name_wc'] = items['item_name'].map(lambda x: len(str(x).split(' ')))
txtFeatures = pd.DataFrame(tfidf.fit_transform(items['item_name']).toarray())
cols = txtFeatures.columns
for i in range(feature_cnt):
    items['item_name_tfidf_' + str(i)] = txtFeatures[cols[i]]
items.head()
feature_cnt = 25
tfidf = feature_extraction.text.TfidfVectorizer(max_features=feature_cnt)
item_cats['item_category_name_len'] = item_cats['item_category_name'].map(len)
item_cats['item_category_name_wc'] = item_cats['item_category_name'].map(lambda x: len(str(x).split(' ')))
txtFeatures = pd.DataFrame(tfidf.fit_transform(item_cats['item_category_name']).toarray())
cols = txtFeatures.columns
for i in range(feature_cnt):
    item_cats['item_category_name_tfidf_' + str(i)] = txtFeatures[cols[i]]
item_cats.head()
feature_cnt = 25
tfidf = feature_extraction.text.TfidfVectorizer(max_features=feature_cnt)
shops['shop_name_len'] = shops['shop_name'].map(len)
shops['shop_name_wc'] = shops['shop_name'].map(lambda x: len(str(x).split(' ')))
txtFeatures = pd.DataFrame(tfidf.fit_transform(shops['shop_name']).toarray())
cols = txtFeatures.columns
for i in range(feature_cnt):
    shops['shop_name_tfidf_' + str(i)] = txtFeatures[cols[i]]
shops.head()
train['date'] = pd.to_datetime(train['date'], format='%d.%m.%Y')
train['month'] = train['date'].dt.month
train['year'] = train['date'].dt.year
train = train.drop(['date', 'item_price'], axis=1)
train = train.groupby([c for c in train.columns if c not in ['item_cnt_day']], as_index=False)[['item_cnt_day']].sum()
train = train.rename(columns={'item_cnt_day': 'item_cnt_month'})
shop_item_monthly_mean = train[['shop_id', 'item_id', 'item_cnt_month']].groupby(['shop_id', 'item_id'], as_index=False)[['item_cnt_month']].mean()
shop_item_monthly_mean = shop_item_monthly_mean.rename(columns={'item_cnt_month': 'item_cnt_month_mean'})
train = pd.merge(train, shop_item_monthly_mean, how='left', on=['shop_id', 'item_id'])
shop_item_prev_month = train[train['date_block_num'] == 33][['shop_id', 'item_id', 'item_cnt_month']]
shop_item_prev_month = shop_item_prev_month.rename(columns={'item_cnt_month': 'item_cnt_prev_month'})
shop_item_prev_month.head()
train = pd.merge(train, shop_item_prev_month, how='left', on=['shop_id', 'item_id']).fillna(0.0)
train = pd.merge(train, items, how='left', on='item_id')
train = pd.merge(train, item_cats, how='left', on='item_category_id')
train = pd.merge(train, shops, how='left', on='shop_id')
train.head()
test['month'] = 11
test['year'] = 2015
test['date_block_num'] = 34
test = pd.merge(test, shop_item_monthly_mean, how='left', on=['shop_id', 'item_id']).fillna(0.0)
test = pd.merge(test, shop_item_prev_month, how='left', on=['shop_id', 'item_id']).fillna(0.0)
test = pd.merge(test, items, how='left', on='item_id')
test = pd.merge(test, item_cats, how='left', on='item_category_id')
test = pd.merge(test, shops, how='left', on='shop_id')
test['item_cnt_month'] = 0.0
test.head()
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
import seaborn as sns

df_all = pd.concat((train, test), axis=0, ignore_index=True)
stores_hm = df_all.pivot_table(index='shop_id', columns='item_category_id', values='item_cnt_month', aggfunc='count', fill_value=0)
(fig, ax) = plt.subplots(figsize=(10, 10))
_ = sns.heatmap(stores_hm, ax=ax, cbar=False)
stores_hm = test.pivot_table(index='shop_id', columns='item_category_id', values='item_cnt_month', aggfunc='count', fill_value=0)
(fig, ax) = plt.subplots(figsize=(10, 10))
_ = sns.heatmap(stores_hm, ax=ax, cbar=False)
for c in ['shop_name', 'item_name', 'item_category_name']:
    lbl = preprocessing.LabelEncoder()