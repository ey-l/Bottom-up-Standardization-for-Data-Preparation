import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
DATA = '_data/input/competitive-data-science-predict-future-sales/'
sales = pd.read_csv(DATA + 'sales_train.csv', parse_dates=['date'], infer_datetime_format=True, dayfirst=True)
sales.head()
test = pd.read_csv(DATA + 'test.csv')
test.head()
df = sales.groupby([sales.date.apply(lambda x: x.strftime('%Y-%m')), 'item_id', 'shop_id']).sum().reset_index()
df = df[['date', 'item_id', 'shop_id', 'item_cnt_day']]
df = df.pivot_table(index=['item_id', 'shop_id'], columns='date', values='item_cnt_day', fill_value=0).reset_index()
df.head()
df_test = pd.merge(test, df, on=['item_id', 'shop_id'], how='left')
df_test = df_test.fillna(0)
df_test.head()
df_test = df_test.drop(labels=['ID', 'shop_id', 'item_id'], axis=1)
df_test.head()
TARGET = '2015-10'
y_train = df_test[TARGET]
X_train = df_test.drop(labels=[TARGET], axis=1)
print(y_train.shape)
print(X_train.shape)
X_train.head()
print(y_train.shape)
print(X_train.shape)
X_test = df_test.drop(labels=['2013-01'], axis=1)
print(X_test.shape)
from lightgbm import LGBMRegressor
model = LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=32, colsample_bytree=0.9497036, subsample=0.8715623, max_depth=8, reg_alpha=0.04, reg_lambda=0.073, min_split_gain=0.0222415, min_child_weight=40)
print('Training time, it is...')