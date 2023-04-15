import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')
import seaborn as sns
import sys
import itertools
import gc
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import csv
from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import datetime
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
cats = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
train['date'] = pd.to_datetime(train['date'], format='%d.%m.%Y')
train['shop_id'] = train['shop_id'].replace({0: 57, 1: 58, 11: 10, 40: 39})
train = train.loc[train.shop_id.isin(test['shop_id'].unique()), :]
train = train[(train['item_price'] > 0) & (train['item_price'] < 50000)]
train = train[(train['item_cnt_day'] > 0) & (train['item_cnt_day'] < 1000)]
shops['city'] = shops['shop_name'].apply(lambda x: x.split()[0])
shops.loc[shops['city'] == '!Якутск', 'city'] = 'Якутск'
label_encoder = LabelEncoder()
shops['city'] = label_encoder.fit_transform(shops['city'])
items['first_sale_date'] = train.groupby('item_id').agg({'date_block_num': 'min'})['date_block_num']
items = items.apply(lambda x: x.fillna(x.median()) if x.dtype != 'O' else x, axis=0)
cats['split'] = cats['item_category_name'].str.split('-')
cats['type'] = cats['split'].map(lambda x: x[0].strip())
cats['type_code'] = LabelEncoder().fit_transform(cats['type'])
cats['subtype'] = cats['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
cats['subtype_code'] = LabelEncoder().fit_transform(cats['subtype'])
cats = cats[['item_category_id', 'type_code', 'subtype_code']]
train = train.merge(shops, on='shop_id', how='left')
train = train.merge(items, on='item_id', how='left')
train = train.merge(cats, on='item_category_id', how='left')
test = test.merge(shops, on='shop_id', how='left')
test = test.merge(items, on='item_id', how='left')
test = test.merge(cats, on='item_category_id', how='left')
test['date_block_num'] = 34
test['date_block_num'] = test['date_block_num'].astype(np.int8)
test['shop_id'] = test.shop_id.astype(np.int8)
test['item_id'] = test.item_id.astype(np.int16)
test = test.drop(columns='ID')
df = pd.concat([train, test])
group = train.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_day': ['sum']})
group.columns = ['item_cnt_month']
group.reset_index(inplace=True)
df = df.merge(group, on=['date_block_num', 'shop_id', 'item_id'], how='left')
df['item_cnt_month'] = df['item_cnt_month'].fillna(0).clip(0, 20).astype(np.float16)
df = df.drop(['date', 'shop_name', 'item_name', 'item_cnt_day'], axis=1)

def add_mean(df, idx_features):
    assert idx_features[0] == 'date_block_num' and len(idx_features) in [2, 3]
    if len(idx_features) == 2:
        feature_name = idx_features[1] + '_mean_sales'
    else:
        feature_name = idx_features[1] + '_' + idx_features[2] + '_mean_sales'
    group = df.groupby(idx_features).agg({'item_cnt_month': 'mean'})
    group = group.reset_index()
    group = group.rename(columns={'item_cnt_month': feature_name})
    df = df.merge(group, on=idx_features, how='left')
    return df
df = add_mean(df=df, idx_features=['date_block_num', 'item_id'])
df = add_mean(df=df, idx_features=['date_block_num', 'shop_id'])
df = add_mean(df=df, idx_features=['date_block_num', 'item_category_id'])
df = add_mean(df=df, idx_features=['date_block_num', 'item_id', 'city'])
df = add_mean(df=df, idx_features=['date_block_num', 'shop_id', 'item_category_id'])
df = add_mean(df=df, idx_features=['date_block_num', 'shop_id', 'subtype_code'])
df['duration_after_first_sale'] = df['date_block_num'] - df['first_sale_date']
df['month'] = df['date_block_num'] % 12
keep_from_month = 2
valid_month = 33
valid = df.loc[df.date_block_num == valid_month, :]
train = df.loc[df.date_block_num < valid_month, :]
train = train[train.date_block_num >= keep_from_month]
X_train = train.drop(columns='item_cnt_month')
y_train = train.item_cnt_month
X_valid = valid.drop(columns='item_cnt_month')
y_valid = valid.item_cnt_month
test = df.drop(columns='item_cnt_month').loc[df.date_block_num == 34, :]
model_lgb = LGBMRegressor(colsample_bytree=0.8, learning_rate=0.01, max_depth=8, min_child_weight=1, min_split_gain=0.0222415, n_estimators=35000, num_leaves=966, reg_alpha=0.04, reg_lambda=0.073, subsample=0.6)
start = datetime.datetime.now()