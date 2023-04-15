import pandas as pd
import numpy as np
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')
from string import ascii_letters
import plotly.express as px
import sys
import gc
import pickle
import time
df_train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
df_test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
df_submission = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
df_shop = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
df_item_cat = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
df_item = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
print(df_train.head())
print('Shape of the data Set is :  ', df_train.shape)
print('Size of the data Set is  :  ', df_train.size)
print(df_train.info())
df_train['date'] = pd.to_datetime(df_train['date'])
print(df_train.info())
plt.figure(figsize=(18, 10))
plt.title('Items Sold Per Month', fontsize=22)
ax = sns.barplot(x='date_block_num', y='item_id', data=df_train, estimator=np.sum)
ax = ax.set(xlabel='Date Blocks', ylabel='Item ID')
plt.xticks(rotation=45)

plt.figure(figsize=(18, 10))
plt.title('Items Sold Shop ID')
ax = sns.barplot(x='shop_id', y='item_id', data=df_train, estimator=np.mean)
ax = ax.set(xlabel='Shop Id', ylabel='Item ID')
plt.xticks(rotation=45)

df_shop.head(2)
df_shop['shop_name']
df_shop['shop_id'].count()
df_shop.loc[df_shop['shop_id'] == 0, 'shop_id'] = 57
df_shop.loc[df_shop['shop_id'] == 1, 'shop_id'] = 58
df_shop.loc[df_shop['shop_id'] == 10, 'shop_id'] = 11
duplicate_shop_names = {df_shop.loc[df_shop['shop_id'] == 57, 'shop_name'].values[0]: df_shop.loc[df_shop['shop_id'] == 57, 'shop_name'].values[1], df_shop.loc[df_shop['shop_id'] == 58, 'shop_name'].values[0]: df_shop.loc[df_shop['shop_id'] == 58, 'shop_name'].values[1], df_shop.loc[df_shop['shop_id'] == 11, 'shop_name'].values[0]: df_shop.loc[df_shop['shop_id'] == 11, 'shop_name'].values[1]}
df_shop = df_shop.drop_duplicates(subset='shop_id')
df_shop['shop_id'].count()
df_shop['city'] = df_shop['shop_name'].str.split(' ').apply(lambda x: x[0])
df_shop.loc[df_shop.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'
df_shop.loc[df_shop['city'] == '!Якутск', 'city'] = 'Якутск'
df_shop.head(2)
plt.figure(figsize=(18, 10))
plt.title('# of stores in the city', fontsize=24)
ax = sns.barplot(x='city', y='shop_id', data=df_shop, estimator=np.sum)
ax = ax.set(xlabel='Stores', ylabel='City')
plt.xticks(rotation=45)

print('Number of unique shops: {}'.format(len(df_shop['shop_id'].unique())))
df_train['shop_id'].nunique()
df_train.loc[df_train['shop_id'] == 0, 'shop_id'] = 57
df_train.loc[df_train['shop_id'] == 1, 'shop_id'] = 58
df_train.loc[df_train['shop_id'] == 10, 'shop_id'] = 11
df_train['shop_id'].nunique()
df_item.head(2)
df_item['item_name'].nunique()
df_item_cat.head(2)
df_item_cat['category'] = df_item_cat['item_category_name'].str.split('-')
df_item_cat['type'] = df_item_cat['category'].apply(lambda x: x[0].strip())
df_item_cat['sub_type'] = df_item_cat['category'].apply(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
df_item_cat.drop(['item_category_name', 'category'], inplace=True, axis=1)
df_item_cat.head(4)
print('    Unique Types: ', df_item_cat['type'].nunique())
print('Unique Sub Types: ', df_item_cat['sub_type'].nunique())
plt.figure(figsize=(18, 10))
plt.title('# of types of items category wise', fontsize=24)
ax = sns.barplot(x='item_category_id', y='type', data=df_item_cat, estimator=np.sum)
ax = ax.set(xlabel='Category ID', ylabel='Type')
plt.xticks(rotation=45)

master_data = pd.merge(df_train, df_item, how='left', on='item_id')
master_data = pd.merge(master_data, df_item_cat, how='left', on='item_category_id')
master_data = pd.merge(master_data, df_shop, how='left', on='shop_id')
print(' Shape: ', master_data.shape)
print('  Size: ', master_data.size)
month = master_data.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_price': 'mean', 'item_cnt_day': 'sum'}).reset_index()
month = pd.merge(month, df_item, how='left', on='item_id')
month = pd.merge(month, df_item_cat, how='left', on='item_category_id')
month = pd.merge(month, df_shop, how='left', on='shop_id')
print(' Shape: ', month.shape)
print('  Size: ', month.size)
month['tota_sales'] = month['item_price'] * month['item_cnt_day']
plt.figure(figsize=(18, 10))
plt.title('# Monthly sales', fontsize=24)
ax = sns.lineplot(x='date_block_num', y='tota_sales', data=month, estimator=np.sum)
ax = ax.set(xlabel='Date Block', ylabel='Total Sales')
plt.xticks(rotation=45)

month['month'] = month['date_block_num'].apply(lambda month: (month + 1) % 12)
month = pd.concat([month, pd.get_dummies(month['shop_id'], drop_first=True, prefix='shop_')], axis=1)
month = pd.concat([month, pd.get_dummies(month['type'], drop_first=True, prefix='type')], axis=1)
month = pd.concat([month, pd.get_dummies(month['sub_type'], drop_first=True, prefix='sub_type')], axis=1)
print(' Shape: ', month.shape)
print('  Size: ', month.size)
shop_col = [col for col in month.columns if 'shop__' in col]
type_col = [col for col in month.columns if 'type_' in col]
sub_type_col = [col for col in month.columns if 'sub_type_' in col]
features = ['month', 'shop_id', 'item_id', 'item_price'] + type_col + sub_type_col
target = ['item_cnt_day']
from sklearn.model_selection import train_test_split
X_feature = month[features].fillna(value=0)
Y_target = month[target].fillna(value=0)
(X_train, X_test, Y_train, Y_test) = train_test_split(X_feature, Y_target, test_size=0.3, random_state=0)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()