import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
sns.set(style='darkgrid')
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
cats = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
print(f'items.csv : {items.shape}')
items.info()
print(f'item_categories.csv : {cats.shape}')
cats.info()
print(f'shops.csv : {shops.shape}')
shops.info()
print(f'sales_train.csv : {train.shape}')
train.info()
print(f'test.csv : {test.shape}')
test.info()
train.isnull().sum()
plt.figure(figsize=(10, 4))
plt.xlim(-100, 3000)
flierprops = dict(marker='o', markerfacecolor='purple', markersize=6, linestyle='none', markeredgecolor='black')
sns.boxplot(x=train.item_cnt_day, flierprops=flierprops)
plt.figure(figsize=(10, 4))
plt.xlim(train.item_price.min(), train.item_price.max() * 1.1)
sns.boxplot(x=train.item_price, flierprops=flierprops)
train = train[(train.item_price < 300000) & (train.item_cnt_day < 1000)]
train = train[train.item_price > 0].reset_index(drop=True)
train.loc[train.item_cnt_day < 1, 'item_cnt_day'] = 0
train.loc[train.shop_id == 0, 'shop_id'] = 57
test.loc[test.shop_id == 0, 'shop_id'] = 57
train.loc[train.shop_id == 1, 'shop_id'] = 58
test.loc[test.shop_id == 1, 'shop_id'] = 58
train.loc[train.shop_id == 10, 'shop_id'] = 11
test.loc[test.shop_id == 10, 'shop_id'] = 11
shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'
shops['city'] = shops.shop_name.str.split(' ').map(lambda x: x[0])
shops['category'] = shops.shop_name.str.split(' ').map(lambda x: x[1])
shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'
category = []
for cat in shops.category.unique():
    if len(shops[shops.category == cat]) >= 5:
        category.append(cat)
shops.category = shops.category.apply(lambda x: x if x in category else 'other')
from sklearn.preprocessing import LabelEncoder
shops['shop_category'] = LabelEncoder().fit_transform(shops.category)
shops['shop_city'] = LabelEncoder().fit_transform(shops.city)
shops = shops[['shop_id', 'shop_category', 'shop_city']]
cats['type_code'] = cats.item_category_name.apply(lambda x: x.split(' ')[0]).astype(str)
cats.loc[(cats.type_code == 'Игровые') | (cats.type_code == 'Аксессуары'), 'category'] = 'Игры'
category = []
for cat in cats.type_code.unique():
    if len(cats[cats.type_code == cat]) >= 5:
        category.append(cat)
cats.type_code = cats.type_code.apply(lambda x: x if x in category else 'etc')
cats.type_code = LabelEncoder().fit_transform(cats.type_code)
cats['split'] = cats.item_category_name.apply(lambda x: x.split('-'))
cats['subtype'] = cats.split.apply(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
cats['subtype_code'] = LabelEncoder().fit_transform(cats['subtype'])
cats = cats[['item_category_id', 'subtype_code', 'type_code']]
import re

def name_correction(x):
    x = x.lower()
    x = x.partition('[')[0]
    x = x.partition('(')[0]
    x = re.sub('[^A-Za-z0-9А-Яа-я]+', ' ', x)
    x = x.replace('  ', ' ')
    x = x.strip()
    return x
(items['name1'], items['name2']) = items.item_name.str.split('[', 1).str
(items['name1'], items['name3']) = items.item_name.str.split('(', 1).str
items['name2'] = items.name2.str.replace('[^A-Za-z0-9А-Яа-я]+', ' ').str.lower()
items['name3'] = items.name3.str.replace('[^A-Za-z0-9А-Яа-я]+', ' ').str.lower()
items = items.fillna('0')
items['item_name'] = items['item_name'].apply(lambda x: name_correction(x))
items.name2 = items.name2.apply(lambda x: x[:-1] if x != '0' else '0')
items['type'] = items.name2.apply(lambda x: x[0:8] if x.split(' ')[0] == 'xbox' else x.split(' ')[0])
items.loc[(items.type == 'x360') | (items.type == 'xbox360') | (items.type == 'xbox 360'), 'type'] = 'xbox 360'
items.loc[items.type == '', 'type'] = 'mac'
items.type = items.type.apply(lambda x: x.replace(' ', ''))
items.loc[(items.type == 'pc') | (items.type == 'pс') | (items.type == 'pc'), 'type'] = 'pc'
items.loc[items.type == 'рs3', 'type'] = 'ps3'
group_sum = items.groupby(['type']).agg({'item_id': 'count'})
group_sum = group_sum.reset_index()
drop_cols = []
for cat in group_sum.type.unique():
    if group_sum.loc[group_sum.type == cat, 'item_id'].values[0] < 40:
        drop_cols.append(cat)
items.name2 = items.name2.apply(lambda x: 'other' if x in drop_cols else x)
items = items.drop(['type'], axis=1)
items.name2 = LabelEncoder().fit_transform(items.name2)
items.name3 = LabelEncoder().fit_transform(items.name3)
items.drop(['item_name', 'name1'], axis=1, inplace=True)
items.head()
from itertools import product
matrix = []
cols = ['date_block_num', 'shop_id', 'item_id']
for i in range(34):
    sales = train[train.date_block_num == i]
    matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype=np.int16))
matrix = pd.DataFrame(np.vstack(matrix), columns=cols)
matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)
matrix['shop_id'] = matrix['shop_id'].astype(np.int8)
matrix['item_id'] = matrix['item_id'].astype(np.int16)
matrix.sort_values(cols, inplace=True)
group = train.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_day': ['sum']})
group.columns = ['item_cnt_month']
group.reset_index(inplace=True)
matrix = pd.merge(matrix, group, on=cols, how='left')
matrix['item_cnt_month'] = matrix['item_cnt_month'].fillna(0).astype(np.float16)
test['date_block_num'] = 34
test['date_block_num'] = test['date_block_num'].astype(np.int8)
test['shop_id'] = test.shop_id.astype(np.int8)
test['item_id'] = test.item_id.astype(np.int16)
matrix = pd.concat([matrix, test.drop(['ID'], axis=1)], ignore_index=True, sort=False, keys=cols)
matrix.fillna(0, inplace=True)
matrix = pd.merge(matrix, shops, on=['shop_id'], how='left')
matrix = pd.merge(matrix, items, on=['item_id'], how='left')
matrix = pd.merge(matrix, cats, on=['item_category_id'], how='left')
matrix['shop_city'] = matrix['shop_city'].astype(np.int8)
matrix['shop_category'] = matrix['shop_category'].astype(np.int8)
matrix['item_category_id'] = matrix['item_category_id'].astype(np.int8)
matrix['subtype_code'] = matrix['subtype_code'].astype(np.int8)
matrix['name2'] = matrix['name2'].astype(np.int8)
matrix['name3'] = matrix['name3'].astype(np.int16)
matrix['type_code'] = matrix['type_code'].astype(np.int8)

def lag_feature(df, lags, cols):
    for col in cols:
        print(col)
        tmp = df[['date_block_num', 'shop_id', 'item_id', col]]
        for i in lags:
            shifted = tmp.copy()
            shifted.columns = ['date_block_num', 'shop_id', 'item_id', col + '_lag_' + str(i)]
            shifted.date_block_num = shifted.date_block_num + i
            df = pd.merge(df, shifted, on=['date_block_num', 'shop_id', 'item_id'], how='left')
    return df
matrix = lag_feature(matrix, [1, 2, 3], ['item_cnt_month'])
group = matrix.groupby(['date_block_num']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_avg_item_cnt']
group.reset_index(inplace=True)
matrix = pd.merge(matrix, group, on=['date_block_num'], how='left')
matrix.date_avg_item_cnt = matrix['date_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], ['date_avg_item_cnt'])
matrix.drop(['date_avg_item_cnt'], axis=1, inplace=True)
group = matrix.groupby(['date_block_num', 'item_id']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_item_avg_item_cnt']
group.reset_index(inplace=True)
matrix = pd.merge(matrix, group, on=['date_block_num', 'item_id'], how='left')
matrix.date_item_avg_item_cnt = matrix['date_item_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], ['date_item_avg_item_cnt'])
matrix.drop(['date_item_avg_item_cnt'], axis=1, inplace=True)
group = train.groupby(['item_id']).agg({'item_price': ['mean']})
group.columns = ['item_avg_item_price']
group.reset_index(inplace=True)
matrix = matrix.merge(group, on=['item_id'], how='left')
matrix['item_avg_item_price'] = matrix.item_avg_item_price.astype(np.float16)
group = train.groupby(['date_block_num', 'item_id']).agg({'item_price': ['mean']})
group.columns = ['date_item_avg_item_price']
group.reset_index(inplace=True)
matrix = matrix.merge(group, on=['date_block_num', 'item_id'], how='left')
matrix['date_item_avg_item_price'] = matrix.date_item_avg_item_price.astype(np.float16)
lags = [1, 2, 3]
matrix = lag_feature(matrix, lags, ['date_item_avg_item_price'])
for i in lags:
    matrix['delta_price_lag_' + str(i)] = (matrix['date_item_avg_item_price_lag_' + str(i)] - matrix['item_avg_item_price']) / matrix['item_avg_item_price']

def select_trends(row):
    for i in lags:
        if row['delta_price_lag_' + str(i)]:
            return row['delta_price_lag_' + str(i)]
    return 0
matrix['delta_price_lag'] = matrix.apply(select_trends, axis=1)
matrix['delta_price_lag'] = matrix.delta_price_lag.astype(np.float16)
matrix['delta_price_lag'].fillna(0, inplace=True)
features_to_drop = ['item_avg_item_price', 'date_item_avg_item_price']
for i in lags:
    features_to_drop.append('date_item_avg_item_price_lag_' + str(i))
    features_to_drop.append('delta_price_lag_' + str(i))
matrix.drop(features_to_drop, axis=1, inplace=True)
matrix['item_shop_first_sale'] = matrix['date_block_num'] - matrix.groupby(['item_id', 'shop_id'])['date_block_num'].transform('min')
matrix['item_first_sale'] = matrix['date_block_num'] - matrix.groupby(['item_id'])['date_block_num'].transform('min')
matrix['month'] = matrix['date_block_num'] % 12
matrix = matrix[matrix['date_block_num'] > 3]
matrix.head()
matrix.info()
import gc
import pickle
from xgboost import XGBRegressor
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = (12, 4)
data = matrix.copy()
del matrix
gc.collect()
data[data['date_block_num'] == 34].shape
data.drop(['item_id'], axis=1, inplace=True)
X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = data[data.date_block_num < 33]['item_cnt_month']
X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = data[data.date_block_num == 33]['item_cnt_month']
X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)
Y_train = Y_train.clip(0, 20)
Y_valid = Y_valid.clip(0, 20)
del group
del train
del cats
del shops
del items
del data
gc.collect()
model = XGBRegressor(objective='reg:squarederror', max_depth=10, n_estimators=600, min_child_weight=0.1, subsample=1, eta=0.1)