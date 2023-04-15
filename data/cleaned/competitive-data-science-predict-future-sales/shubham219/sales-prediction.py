import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns
import pickle
import lightgbm as lgb
from itertools import product
item_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
item_cat_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
sales_train_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
shops_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
test_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
print('Shape of the Items: ', item_df.shape)

print('No of Unique Item Id: ', item_df['item_id'].nunique())
print('Any Null Values?')
print(item_df.isnull().sum())
print('Shape of the Items: ', item_cat_df.shape)

print('No of Unique Item Category: ', item_cat_df['item_category_id'].nunique())
print('Any Null Values?')
print(item_cat_df.isnull().sum())
print('Shape of the Items: ', sales_train_df.shape)
print('\n')

print('No of Unique date block num: ', sales_train_df['date_block_num'].nunique())
print('No of Unique shop id: ', sales_train_df['shop_id'].nunique())
print('No of Unique item id: ', sales_train_df['item_id'].nunique())
print('Any Null Values?')
print(sales_train_df.isnull().sum())
print('\nBasis Stats of Item Price')

print('\nBasic Stats of Item_count_day')

print('\nThere are few negative values too in the sale and count does they represent any return? How many such values are there?')


print('\nStats of negative item count day')

median = sales_train_df.loc[(sales_train_df['shop_id'] == 32) & (sales_train_df['item_id'] == 2973), 'item_price'].median()
print('Median Values: ', median)
sales_train_df.loc[(sales_train_df['shop_id'] == 32) & (sales_train_df['item_price'] == -1), 'item_price'] = median
print('Is there any problem with only one shop?')
print(sales_train_df.loc[sales_train_df['item_cnt_day'] < 0, 'shop_id'].nunique())
print('\nIs there only item having negative count?')
print(sales_train_df.loc[sales_train_df['item_cnt_day'] < 0, 'item_id'].nunique())
print('\n Are negative values coming every month?')
print(sales_train_df.loc[sales_train_df['item_cnt_day'] < 0, 'date_block_num'].unique())
print('\nEvery month negative values are found. Are they return items?')
sales_train_df.loc[sales_train_df.shop_id == 0, 'shop_id'] = 57
test_df.loc[test_df.shop_id == 0, 'shop_id'] = 57
sales_train_df.loc[sales_train_df.shop_id == 1, 'shop_id'] = 58
test_df.loc[test_df.shop_id == 1, 'shop_id'] = 58
sales_train_df.loc[sales_train_df.shop_id == 10, 'shop_id'] = 11
test_df.loc[test_df.shop_id == 10, 'shop_id'] = 11
from sklearn.preprocessing import LabelEncoder
shops_df.loc[shops_df.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'
shops_df['city'] = shops_df['shop_name'].str.split(' ').map(lambda x: x[0])
shops_df.loc[shops_df.city == '!Якутск', 'city'] = 'Якутск'
shops_df['city_code'] = LabelEncoder().fit_transform(shops_df['city'])
shops_df = shops_df[['shop_id', 'city_code']]
item_cat_df['split'] = item_cat_df['item_category_name'].str.split('-')
item_cat_df['type'] = item_cat_df['split'].map(lambda x: x[0].strip())
item_cat_df['type_code'] = LabelEncoder().fit_transform(item_cat_df['type'])
item_cat_df['subtype'] = item_cat_df['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
item_cat_df['subtype_code'] = LabelEncoder().fit_transform(item_cat_df['subtype'])
item_cat_df = item_cat_df[['item_category_id', 'type_code', 'subtype_code']]
item_df.drop(['item_name'], axis=1, inplace=True)
train_shops_item = sales_train_df['shop_id'].astype(str) + '_' + sales_train_df['item_id'].astype(str)
train_shops_item = train_shops_item.drop_duplicates().reset_index(drop=True)
train_shops_item = sales_train_df['item_id']
test_shops_item = test_df['shop_id'].astype(str) + '_' + test_df['item_id'].astype(str)
test_shops_item = test_shops_item.drop_duplicates().reset_index(drop=True)
test_shops_item = test_df['item_id']
print('Total Numbers of Items Not In Train')
len(set(test_shops_item).difference(train_shops_item))
plt.figure(figsize=(10, 4), dpi=80)
plt.title('Item Count')
sns.boxplot(x=sales_train_df['item_cnt_day'])

plt.figure(figsize=(10, 4), dpi=80)
plt.title('Looking At Outlier')
plt.xlim((2000, 2500))
sns.boxplot(x=sales_train_df['item_cnt_day'])

plt.figure(figsize=(10, 4), dpi=80)
plt.title('Item Price')
sns.boxplot(x=sales_train_df['item_price'])


def remove_outliers(df):
    df = df[df['item_price'] <= 100000]
    df = df[df['item_cnt_day'] <= 1500]
    return df
sales_train_df = remove_outliers(sales_train_df)
print('Check After Outliers Removal')
plt.figure(figsize=(10, 4), dpi=80)
plt.title('Item Count')
sns.boxplot(x=sales_train_df['item_cnt_day'])

plt.figure(figsize=(10, 4), dpi=80)
plt.title('Item Price')
sns.boxplot(x=sales_train_df['item_price'])

sales_train_df['date'] = pd.to_datetime(sales_train_df['date'], format='%d.%m.%Y')
tmp = test_df.copy()
tmp['date_block_num'] = 34
tmp.drop('ID', axis=1, inplace=True)
tmp = pd.concat([tmp, sales_train_df[['shop_id', 'item_id', 'date_block_num']]], axis=0, ignore_index=True)
matrix = []
cols = ['date_block_num', 'shop_id', 'item_id']
for i in range(35):
    sales = tmp[tmp['date_block_num'] == i]
    matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='int16'))
matrix = pd.DataFrame(np.vstack(matrix), columns=cols)
matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)
matrix['shop_id'] = matrix['shop_id'].astype(np.int8)
matrix['item_id'] = matrix['item_id'].astype(np.int16)
matrix.sort_values(cols, inplace=True)
print(matrix.shape)
train_df = sales_train_df.groupby(cols).agg({'item_cnt_day': ['sum', 'mean', 'median']}).clip(0, 20)
train_df.columns = ['item_cnt_month', 'avg_item_cnt_month', 'median_item_cnt_month']
train_df.reset_index(inplace=True)
group = sales_train_df.groupby(cols).agg({'item_price': 'mean'})
train_df = pd.merge(train_df, group, on=cols, how='left')
train_df = pd.merge(matrix, train_df, on=cols, how='left')
train_df.rename(columns={'item_price': 'avg_item_price'}, inplace=True)
print('Shape of the train data: ', train_df.shape)
train_df.head()
item_avg_price_df = sales_train_df.groupby(['item_id'])['item_price'].mean().reset_index()
train_df = pd.merge(train_df, item_avg_price_df, how='left', on='item_id')
train_df['avg_item_price'] = np.where(train_df['avg_item_price'].isnull(), train_df['item_price'], train_df['avg_item_price'])
train_df.drop('item_price', axis=1, inplace=True)
train_df.head()
train_df = pd.merge(train_df, item_df, how='left', on='item_id')
train_df = pd.merge(train_df, item_cat_df, how='left', on='item_category_id')
train_df = pd.merge(train_df, shops_df, how='left', on='shop_id')
train_df.head()
cols = ['shop_id', 'item_id', 'date_block_num', 'item_cnt_month']
cols_to_merge = ['shop_id', 'item_id', 'date_block_num']
lags = [1, 3, 6, 12, 18]
for i in lags:
    print(i)
    shifted_df = train_df[cols].copy()
    shifted_df['date_block_num'] = shifted_df['date_block_num'] + i
    shifted_df[f'item_cnt_month_lag_{i}'] = shifted_df['item_cnt_month']
    shifted_df.drop('item_cnt_month', axis=1, inplace=True)
    train_df = pd.merge(train_df, shifted_df, on=cols_to_merge, how='left')
train_df = train_df.fillna(0)
train_df = train_df.fillna(0)
train_df[(train_df['shop_id'] == 45) & (train_df['item_id'] == 969)]
(train_df['shop_id'].nunique(), train_df['item_id'].nunique(), train_df['item_category_id'].nunique())
test = train_df[train_df['date_block_num'] == 34]
test.drop('item_cnt_month', axis=1, inplace=True)
valid_dataset = train_df[train_df['date_block_num'] == 33].copy()
valid_target = valid_dataset['item_cnt_month']
valid_dataset.drop('item_cnt_month', axis=1, inplace=True)
train_dataset = train_df[train_df['date_block_num'] < 33].copy()
train_target = train_dataset['item_cnt_month']
train_dataset.drop('item_cnt_month', axis=1, inplace=True)
(test.shape, valid_dataset.shape, valid_target.shape, train_dataset.shape, train_target.shape)
lgb_params = {'metric': {'rmse'}, 'num_leaves': 12, 'learning_rate': 0.02, 'feature_fraction': 0.8, 'max_depth': 5, 'verbose': 0, 'num_boost_round': 1000, 'early_stopping_rounds': 100, 'nthread': -1}
lgbtrain = lgb.Dataset(data=train_dataset, label=train_target)
lgbval = lgb.Dataset(data=valid_dataset, label=valid_target, reference=lgbtrain)
model = lgb.train(lgb_params, lgbtrain, valid_sets=[lgbtrain, lgbval], num_boost_round=lgb_params['num_boost_round'], early_stopping_rounds=lgb_params['early_stopping_rounds'], verbose_eval=100)
from sklearn.metrics import mean_squared_error
pred = model.predict(valid_dataset)
rmse_score = np.sqrt(mean_squared_error(valid_target, pred))
print('Rmse Score: ', rmse_score)
pred_test = pd.DataFrame(model.predict(test), columns=['item_cnt_month'])
pred_test = pd.concat([test_df, pred_test], axis=1)
pred_test = pred_test[['ID', 'item_cnt_month']]
