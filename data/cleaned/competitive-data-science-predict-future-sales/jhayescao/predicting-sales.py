import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import sys
import gc
import pickle
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
train['date'] = pd.to_datetime(train['date'])
train.head()
test.head()
df = train.groupby([train.date.apply(lambda x: x.strftime('%Y-%m')), 'item_id', 'shop_id']).sum().reset_index()
df = df[['date', 'item_id', 'shop_id', 'item_cnt_day']]
df = df.pivot_table(index=['item_id', 'shop_id'], columns='date', values='item_cnt_day', fill_value=0).reset_index()
df.head()
test_df = pd.merge(test, df, on=['item_id', 'shop_id'], how='left')
test_df = test_df.fillna(0)
test_df.head()
df_test = test_df.drop(labels=['ID', 'shop_id', 'item_id'], axis=1)
df_test.head()
PRED = '2015-11'
y_train = df_test[PRED]
X_train = df_test.drop(labels=[PRED], axis=1)
print(y_train.shape)
print(X_train.shape)
X_train.head()
X_test = df_test.drop(labels=['2013-01'], axis=1)
print(X_test.shape)
from lightgbm import LGBMRegressor
model = LGBMRegressor(n_estimators=1000, learning_rate=0.03, num_leaves=32, colsample_bytree=0.9497036, subsample=0.8715623, max_depth=8, reg_alpha=0.04, reg_lambda=0.073, min_split_gain=0.0222415, min_child_weight=40)