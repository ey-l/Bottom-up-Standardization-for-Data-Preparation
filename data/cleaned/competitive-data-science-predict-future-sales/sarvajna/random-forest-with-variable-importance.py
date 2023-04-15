import numpy as np
import pandas as pd
import sklearn
import nltk
import datetime
train_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
submission_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
items_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
item_categories_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
shops_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
print('Shape of train data : {}, Shape of test data : {}'.format(train_df.shape, test_df.shape))
[a for a in train_df.columns if a not in test_df]
train_df.head()
test_df.head()
items_df.head()
items_df.describe()
feature_count = 25
items_df['item_name_length'] = items_df['item_name'].map(lambda x: len(x))
items_df['item_name_word_count'] = items_df['item_name'].map(lambda x: len(x.split(' ')))
tfidf = sklearn.feature_extraction.text.TfidfVectorizer(max_features=feature_count)
items_df_item_name_text_features = pd.DataFrame(tfidf.fit_transform(items_df['item_name']).toarray())
print('Shape of items_df_item_name_text_features : {}'.format(items_df_item_name_text_features.shape))
cols = items_df_item_name_text_features.columns
for idx in range(feature_count):
    items_df['item_name_tfidf_' + str(idx)] = items_df_item_name_text_features[cols[idx]]
items_df.head()
item_categories_df.head()
item_categories_df.describe()
feature_count = 25
item_categories_df['item_categories_name_length'] = item_categories_df['item_category_name'].map(lambda x: len(x))
item_categories_df['item_categories_name_word_count'] = item_categories_df['item_category_name'].map(lambda x: len(x.split(' ')))
tfidf = sklearn.feature_extraction.text.TfidfVectorizer(max_features=feature_count)
item_categories_df_item_category_name_text_features = pd.DataFrame(tfidf.fit_transform(item_categories_df['item_category_name']).toarray())
cols = item_categories_df_item_category_name_text_features.columns
for idx in range(feature_count):
    item_categories_df['item_category_name_tfidf_' + str(idx)] = item_categories_df_item_category_name_text_features[cols[idx]]
item_categories_df.head()
shops_df.tail()
shops_df.describe()
feature_count = 25
shops_df['shop_name_length'] = shops_df['shop_name'].map(lambda x: len(x))
shops_df['shop_name_word_count'] = shops_df['shop_name'].map(lambda x: len(x.split(' ')))
tfidf = sklearn.feature_extraction.text.TfidfVectorizer(max_features=feature_count)
shops_df_shop_name_text_features = pd.DataFrame(tfidf.fit_transform(shops_df['shop_name']).toarray())
cols = shops_df_shop_name_text_features.columns
for idx in range(feature_count):
    shops_df['shop_name_tfidf_' + str(idx)] = shops_df_shop_name_text_features[cols[idx]]
shops_df.head()
train_df.head()
train_df['date'] = pd.to_datetime(train_df['date'], format='%d.%m.%Y')
train_df['month'] = train_df['date'].dt.month
train_df['year'] = train_df['date'].dt.year
train_df = train_df.drop(['date', 'item_price'], axis=1)
train_df = train_df.groupby([c for c in train_df.columns if c not in ['item_cnt_day']], as_index=False)[['item_cnt_day']].sum()
train_df = train_df.rename(columns={'item_cnt_day': 'item_cnt_month'})
shop_item_monthly_mean = train_df[['shop_id', 'item_id', 'item_cnt_month']].groupby(['shop_id', 'item_id'], as_index=False)[['item_cnt_month']].mean()
shop_item_monthly_mean = shop_item_monthly_mean.rename(columns={'item_cnt_month': 'item_cnt_month_mean'})
train_df = pd.merge(train_df, shop_item_monthly_mean, how='left', on=['shop_id', 'item_id'])
train_df.head()
shop_item_prev_month = train_df[train_df['date_block_num'] == 33][['shop_id', 'item_id', 'item_cnt_month']]
shop_item_prev_month = shop_item_prev_month.rename(columns={'item_cnt_month': 'item_cnt_prev_month'})
shop_item_prev_month.head()
train_df = pd.merge(train_df, shop_item_prev_month, how='left', on=['shop_id', 'item_id'])
train_df.head()
np.where(pd.isnull(train_df))
train_df = train_df.fillna(0.0)
train_df.head()
train_df = pd.merge(train_df, items_df, how='left', on='item_id')
train_df.head()
train_df = pd.merge(train_df, item_categories_df, how='left', on=['item_category_id'])
train_df.head()
train_df = pd.merge(train_df, shops_df, how='left', on=['shop_id'])
train_df.head()
test_df['month'] = 11
test_df['year'] = 2015
test_df['date_block_num'] = 34
test_df.head()
shop_item_monthly_mean.head()
len(test_df)
len(train_df)
test_df = pd.merge(test_df, shop_item_monthly_mean, how='left', on=['shop_id', 'item_id'])
print(len(test_df))
test_df.head()
5320 in train_df.item_id.values
5233 in train_df.item_id.values
test_df = pd.merge(test_df, shop_item_prev_month, how='left', on=['shop_id', 'item_id'])
test_df.head()
test_df = pd.merge(test_df, items_df, how='left', on='item_id')
test_df.head()
test_df = pd.merge(test_df, item_categories_df, how='left', on='item_category_id')
test_df = pd.merge(test_df, shops_df, how='left', on='shop_id')
test_df = test_df.fillna(0.0)
test_df['item_cnt_month'] = 0.0
test_df.head()
import matplotlib.pyplot as plt
import seaborn as sns

train_test_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
print('train_df.shape = {}, test_df.shape = {}, train_test_df.shape = {}'.format(train_df.shape, test_df.shape, train_test_df.shape))
stores_hm = train_test_df.pivot_table(index='shop_id', columns='item_category_id', values='item_cnt_month', aggfunc='count', fill_value=0)
print('stores_hm.shape = {}'.format(stores_hm.shape))
stores_hm.tail()
(fig, ax) = plt.subplots(figsize=(15, 15))
sns.heatmap(stores_hm, ax=ax, cbar=False)
stores_hm = train_df.pivot_table(index='shop_id', columns='item_category_id', values='item_cnt_month', aggfunc='count', fill_value=0)
(_, ax) = plt.subplots(figsize=(15, 15))
sns.heatmap(stores_hm, ax=ax, cbar=False)
stores_hm = test_df.pivot_table(index='shop_id', columns='item_category_id', values='item_cnt_month', aggfunc='count', fill_value=0)
(_, ax) = plt.subplots(figsize=(15, 15))
sns.heatmap(stores_hm, ax=ax, cbar=False)
for c in ['shop_name', 'item_category_name', 'item_name']:
    le = sklearn.preprocessing.LabelEncoder()