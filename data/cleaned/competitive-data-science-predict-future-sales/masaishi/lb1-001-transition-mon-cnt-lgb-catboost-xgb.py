
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import lightgbm as lgb
import catboost
from xgboost import XGBRegressor
from xgboost import plot_importance

def plot_features(booster, figsize):
    (fig, ax) = plt.subplots(1, 1, figsize=figsize)
    return plot_importance(booster=booster, ax=ax)
from tqdm import tqdm
from itertools import product
from sklearn.model_selection import train_test_split
import seaborn as sns
import sys
import os
import gc
from glob import glob
import pickle
import json
import subprocess
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold, train_test_split, RepeatedStratifiedKFold
DF_CEILING_VALUE = 20000.0
SHOP_CEILING_VALUE = 999999.9
ITEM_CEILING_VALUE = 999999.9
PATH = '_data/input/competitive-data-science-predict-future-sales/'
df = pd.read_csv(PATH + 'sales_train.csv')
df_test = pd.read_csv(PATH + 'test.csv')
sample = pd.read_csv(PATH + 'sample_submission.csv')
items = pd.read_csv(PATH + 'items.csv')
shops = pd.read_csv(PATH + 'shops.csv')
item_cats = pd.read_csv(PATH + 'item_categories.csv')
data_files_names = ['df', 'df_test', 'sample', 'items', 'shops', 'item_cats']
data_files = [df, df_test, sample, items, shops, item_cats]
df_test = df_test.drop('ID', axis=1)
df
df.loc[df['item_cnt_day'] < 0.0, 'item_cnt_day'] = 0.0
df[df['item_cnt_day'] > 100.0]
df.loc[df['item_cnt_day'] > 100.0, 'item_cnt_day'] = 1.0
len(df[df['item_cnt_day'] > 10.0])
df[df['item_cnt_day'] > 10.0]
df.loc[df['shop_id'] == 57, 'shop_id'] = 0
df.loc[df['shop_id'] == 58, 'shop_id'] = 1
df.loc[df['shop_id'] == 11, 'shop_id'] = 10
df.loc[df['shop_id'] == 40, 'shop_id'] = 39
df_test.loc[df_test['shop_id'] == 57, 'shop_id'] = 0
df_test.loc[df_test['shop_id'] == 58, 'shop_id'] = 1
df_test.loc[df_test['shop_id'] == 11, 'shop_id'] = 10
df_test.loc[df_test['shop_id'] == 40, 'shop_id'] = 39
df['shop_item_id'] = df['shop_id'].astype('str').str.zfill(2) + df['item_id'].astype('str').str.zfill(5)
df['item_category_id'] = pd.merge(df, items, on='item_id', how='left')['item_category_id']
df
df = df.groupby(['date_block_num', 'shop_id', 'item_id', 'shop_item_id', 'item_category_id'], as_index=False).agg({'item_cnt_day': 'sum'}).rename(columns={'item_cnt_day': 'mon_shop_item_cnt'})
df
df_test['shop_item_id'] = df_test['shop_id'].astype('str').str.zfill(2) + df_test['item_id'].astype('str').str.zfill(5)
df_test['item_category_id'] = pd.merge(df_test, items, on='item_id', how='left')['item_category_id']
df_test
count = 0
df_ids = df['shop_item_id'].unique()
repeat_count = 0
for one_id in df_test['shop_item_id'].sample(1000):
    if one_id in df_ids:
        count += 1
    repeat_count += 1
    if repeat_count > 1000:
        break
print(count / repeat_count)
plt.figure(figsize=(12, 6))
plt.hist(df['mon_shop_item_cnt'])
df.loc[df['mon_shop_item_cnt'] > 100.0, 'mon_shop_item_cnt'] = 101.0
print(len(df.loc[df['mon_shop_item_cnt'] > 100.0]), len(df.loc[df['mon_shop_item_cnt'] > 101.0]))
len(np.unique(np.concatenate([df['shop_item_id'], df_test['shop_item_id']])))
transition = pd.DataFrame(np.unique(np.concatenate([df['shop_item_id'], df_test['shop_item_id']])), columns=['shop_item_id'])
for i in range(34):
    transition = pd.merge(transition, df[df['date_block_num'] == i].drop(['date_block_num', 'shop_id', 'item_id', 'item_category_id'], axis=1).rename(columns={'mon_shop_item_cnt': i}), on='shop_item_id', how='left')
transition = transition.fillna(0)
transition
plt.figure(figsize=(12, 6))
plt.bar(transition.loc[:, 0:].columns, transition.loc[:, 0:].sum())
plt.figure(figsize=(12, 6))
plt.hist(transition.loc[:, 0:].T.sum())
transition[transition.loc[:, 0:].T.sum() >= 500.0]
transition_max = transition.loc[:, 0:].max().max()
transition_max
std_transition = transition.copy()
std_transition.loc[:, 0:] = std_transition.loc[:, 0:] / transition_max
std_transition
shops.loc[shops['shop_id'] == 57, 'shop_id'] = 0
shops.loc[shops['shop_id'] == 58, 'shop_id'] = 1
shops.loc[shops['shop_id'] == 11, 'shop_id'] = 10
shops.loc[shops['shop_id'] == 40, 'shop_id'] = 39
shops
shop_df = df.groupby(['date_block_num', 'shop_id'], as_index=False).agg({'mon_shop_item_cnt': 'sum'})
shop_transition = pd.DataFrame(shops['shop_id'].unique(), columns=['shop_id'])
for i in range(34):
    shop_transition = pd.merge(shop_transition, shop_df[shop_df['date_block_num'] == i].drop('date_block_num', axis=1).rename(columns={'mon_shop_item_cnt': i}), on='shop_id', how='left')
shop_transition = shop_transition.fillna(0)
shop_transition_max = shop_transition.loc[:, 0:].max().max()
shop_transition.loc[:, 0:] = shop_transition.loc[:, 0:] / shop_transition_max
shop_transition
shop_feature = transition.loc[:, ['shop_item_id']].copy()
shop_feature['shop_id'] = shop_feature['shop_item_id'].str[:2].astype(int)
shop_feature
shop_feature = pd.merge(shop_feature, shop_transition, on='shop_id', how='left')
shop_feature = shop_feature.drop('shop_id', axis=1)
shop_feature
items
item_df = df.groupby(['date_block_num', 'item_id'], as_index=False).agg({'mon_shop_item_cnt': 'sum'})
item_transition = pd.DataFrame(items['item_id'].unique(), columns=['item_id'])
for i in range(34):
    item_transition = pd.merge(item_transition, item_df[item_df['date_block_num'] == i].drop('date_block_num', axis=1).rename(columns={'mon_shop_item_cnt': i}), on='item_id', how='left')
item_transition = item_transition.fillna(0)
item_transition_max = item_transition.loc[:, 0:].max().max()
item_transition.loc[:, 0:] = item_transition.loc[:, 0:] / item_transition_max
item_transition
item_feature = transition.loc[:, ['shop_item_id']].copy()
item_feature['item_id'] = item_feature['shop_item_id'].str[2:].astype(int)
item_feature = pd.merge(item_feature, item_transition, on='item_id', how='left')
item_feature = item_feature.drop('item_id', axis=1)
item_feature
cats_df = df.groupby(['date_block_num', 'item_category_id'], as_index=False).agg({'mon_shop_item_cnt': 'sum'})
cats_transition = pd.DataFrame(cats_df['item_category_id'].unique(), columns=['item_category_id'])
for i in range(34):
    cats_transition = pd.merge(cats_transition, cats_df[cats_df['date_block_num'] == i].drop('date_block_num', axis=1).rename(columns={'mon_shop_item_cnt': i}), on='item_category_id', how='left')
cats_transition = cats_transition.fillna(0)
cats_transition_max = cats_transition.loc[:, 0:].max().max()
cats_transition.loc[:, 0:] = cats_transition.loc[:, 0:] / cats_transition_max
cats_transition
cats_feature = transition.loc[:, ['shop_item_id']].copy()
cats_feature['item_id'] = cats_feature['shop_item_id'].str[2:].astype(int)
cats_feature['item_category_id'] = pd.merge(cats_feature, items, on='item_id', how='left')['item_category_id']
cats_feature = pd.merge(cats_feature, cats_transition, on='item_category_id', how='left')
cats_feature = cats_feature.drop(['item_id', 'item_category_id'], axis=1)
cats_feature
print(shop_feature.loc[:, 0:].mean().mean(), item_feature.loc[:, 0:].mean().mean())
shop_feature.loc[:, 0:] += np.random.normal(0, shop_feature.loc[:, 0:].mean().mean() * 0.025, shop_feature.loc[:, 0:].shape)
item_feature.loc[:, 0:] += np.random.normal(0, item_feature.loc[:, 0:].mean().mean() * 0.025, item_feature.loc[:, 0:].shape)
cats_feature.loc[:, 0:] += np.random.normal(0, cats_feature.loc[:, 0:].mean().mean() * 0.025, cats_feature.loc[:, 0:].shape)
shop_feature
features = [std_transition, shop_feature, item_feature, cats_feature]
del shop_feature, item_feature, cats_feature
gc.collect()
for i in range(len(features)):
    month_means = features[i].loc[:, 0:].mean()
    plt.figure(figsize=(12, 6))
    plt.bar(features[i].loc[:, 0:].columns, month_means)
    x = np.array(features[i].loc[:, 0:].columns)
    x = x.reshape(-1, 1)
    y = np.array(month_means)
    y = y.reshape(-1, 1)
    mm_reg_model = LinearRegression()