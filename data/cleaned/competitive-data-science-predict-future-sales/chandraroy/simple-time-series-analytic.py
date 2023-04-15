import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
cats = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv').set_index('ID')
cats.head()
train.head()
test.head()
train.shape
test.shape
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
ax.set(xlabel='Month', ylabel='Sales')
ax.plot(train.date_block_num, train.item_cnt_day)

len(set(test.item_id)) - len(set(test.item_id).intersection(set(train.item_id)))
train.columns
train[train.date_block_num == 0]
from itertools import product
matrix = []
cols = ['date_block_num', 'shop_id', 'item_id']
for i in range(34):
    sales = train[train.date_block_num == i]
    matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='int16'))
matrix
matrix = pd.DataFrame(np.vstack(matrix), columns=cols)
matrix.head()
matrix.shape
matrix[matrix.date_block_num == 1].shape
matrix[matrix.date_block_num == 3].shape
matrix.info()
matrix.sort_values(cols, inplace=True)
train['revenue'] = train['item_price'] * train['item_cnt_day']
group = train.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_day': ['sum'], 'item_price': ['mean']})
group.columns = ['item_cnt_month', 'avg_item_price_shopwise']
group.reset_index(inplace=True)
group.head()
matrix = pd.merge(matrix, group, on=cols, how='left')
matrix.shape
matrix['item_cnt_month'] = matrix['item_cnt_month'].fillna(0).clip(0, 20).astype(np.float16)
matrix.shape
test['date_block_num'] = 34
test.columns
test['date_block_num'] = test['date_block_num'].astype(np.int8)
test['shop_id'] = test['shop_id'].astype(np.int8)
test['item_id'] = test['item_id'].astype(np.int16)
test.head()
test.shape
cols = ['shop_id', 'item_id', 'date_block_num']
matrix_tmp = matrix.groupby(['shop_id', 'item_id']).mean()['avg_item_price_shopwise'].reset_index()
test = pd.merge(test, matrix_tmp, on=['shop_id', 'item_id'], how='left')
test.shape
test.head()
test.isna().sum()
test['avg_item_price_shopwise'].fillna(0, inplace=True)
test.shape
matrix = pd.concat([matrix, test], ignore_index=True, sort=False, keys=cols)
matrix.fillna(0, inplace=True)
matrix.tail()
matrix.head()
matrix.shape

def lag_feature(df, lags, col):
    tmp = df[['date_block_num', 'shop_id', 'item_id', col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num', 'shop_id', 'item_id', col + '_lag_' + str(i)]
        shifted['date_block_num'] += i
        df = pd.merge(df, shifted, on=['date_block_num', 'shop_id', 'item_id'], how='left')
        return df
matrix = lag_feature(matrix, [1, 2, 3, 6, 12], 'item_cnt_month')
matrix.head()
matrix.shape

def fill_na(df):
    for col in df.columns:
        if ('_lag_' in col) & df[col].isnull().any():
            if 'item_cnt' in col:
                df[col].fillna(0, inplace=True)
    return df
matrix = fill_na(matrix)
df = matrix.copy()
X_train = df[df.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = df[df.date_block_num < 33]['item_cnt_month']
X_validate = df[df.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_validate = df[df.date_block_num == 33]['item_cnt_month']
X_test = df[df.date_block_num == 34].drop(['item_cnt_month'], axis=1)
from xgboost import XGBRegressor
from xgboost import plot_importance
model = XGBRegressor(max_depth=8, n_estimators=1000, min_child_weight=300, colsample_bytree=0.8, subsample=0.8, eta=0.1, seed=42)