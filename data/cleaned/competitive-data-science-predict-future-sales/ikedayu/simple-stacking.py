

import os, sys, re, datetime, gc
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import feature_extraction
from sklearn import preprocessing
from tqdm import tqdm_notebook
pd.set_option('display.max_rows', 600)
pd.set_option('display.max_columns', 50)
for p in [np, pd, sklearn, lgb]:
    print(p.__name__, p.__version__)

def downcast_dtypes(df):
    """
        Changes column types in the dataframe: 
                
                `float64` type to `float32`
                `int64`   type to `int32`
    """
    float_cols = [c for c in df if df[c].dtype == 'float64']
    int_cols = [c for c in df if df[c].dtype == 'int64']
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int32)
    return df

def evaluate(true, pred):
    return np.sqrt(mean_squared_error(true.clip(0.0, 20.0), pred.clip(0.0, 20.0)))
random_seed = 1021
lgb_params = {'feature_fraction': 0.75, 'metric': 'rmse', 'nthread': -1, 'min_data_in_leaf': 2 ** 7, 'bagging_fraction': 0.75, 'learning_rate': 0.03, 'objective': 'mse', 'num_leaves': 2 ** 7, 'bagging_freq': 1, 'verbose': 0}
xgb_params = {'eta': 0.2, 'max_depth': 4, 'objective': 'reg:linear', 'eval_metric': 'rmse', 'seed': random_seed, 'silent': True}
cat_params = {'iterations': 100, 'learning_rate': 0.2, 'depth': 7, 'loss_function': 'RMSE', 'eval_metric': 'RMSE', 'random_seed': random_seed, 'od_type': 'Iter', 'od_wait': 20}
data_dir = Path('_data/input/competitive-data-science-predict-future-sales')
train = pd.read_csv(data_dir / 'sales_train.csv')
test = pd.read_csv(data_dir / 'test.csv')
items = pd.read_csv(data_dir / 'items.csv')
item_cats = pd.read_csv(data_dir / 'item_categories.csv')
shops = pd.read_csv(data_dir / 'shops.csv')
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
sales = pd.concat([train, test.drop(columns='ID')], axis=0, sort=False)
for c in ['shop_name', 'item_name', 'item_category_name']:
    lbl = preprocessing.LabelEncoder()