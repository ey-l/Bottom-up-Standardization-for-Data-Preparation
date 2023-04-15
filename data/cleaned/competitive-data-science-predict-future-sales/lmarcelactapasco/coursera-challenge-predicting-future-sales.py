import numpy as np
import pandas as pd
import os
import scipy.sparse
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from pandas.plotting import scatter_matrix
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

ROOT_FOLDER = '_data/input/competitive-data-science-predict-future-sales/'
df_items = pd.read_csv(os.path.join(ROOT_FOLDER, 'items.csv'))
df_item_categories = pd.read_csv(os.path.join(ROOT_FOLDER, 'item_categories.csv'))
df_sales_train = pd.read_csv(os.path.join(ROOT_FOLDER, 'sales_train.csv'))
df_shops = pd.read_csv(os.path.join(ROOT_FOLDER, 'shops.csv'))
df_test = pd.read_csv(os.path.join(ROOT_FOLDER, 'test.csv'))
print(' Dataset Items ')
df_items.head(1)
print(' Dataset item_categories ')
df_item_categories.head(1)
print(' Dataset sales_train ')
df_sales_train.head(1)
print(' Dataset test ')
df_test.head(1)
print(' Dataset shops ')
df_shops.head(1)
print('Train min/max date: %s / %s' % (df_sales_train.date_block_num.min(), df_sales_train.date_block_num.max()))
print('Sales Train shape: %d rows' % df_sales_train.shape[0])
print('Test: %d rows ' % df_test.shape[0])
df_sales_train.isnull().sum(axis=0).head(15)
df_sales_train.isnull().sum(axis=1).head(15)
index_cols = ['shop_id', 'item_id', 'item_price', 'date_block_num']
gb_train = df_sales_train.groupby(index_cols, as_index=False).agg({'item_cnt_day': 'sum'}, dtype='int32')
gb_train = gb_train.rename(columns={'item_cnt_day': 'item_cnt_month'})
gb_train = gb_train[['shop_id', 'item_id', 'item_price', 'date_block_num', 'item_cnt_month']]
gb_train.fillna(0, inplace=True)
gb_train.head(2)
gb_train[gb_train.item_cnt_month == 0]
gb_train[(gb_train.shop_id == 2) & (gb_train.item_id == 835)]
gb_train.count()
(figure, axe) = plt.subplots(figsize=(12, 12))
axe.set_title(' EDA Item Price VS  Sales Day', weight='bold')
plot = plt.scatter(gb_train.item_price, gb_train.item_cnt_month, marker='o', c='yellow', edgecolor='black', s=30, cmap='viridis', linewidth=0.5)
plt.xlabel('Item Price')
plt.ylabel('Sales Day')
PRICE_OUT = 300000
SALES_OUT = 2000
gb_train = gb_train[(gb_train.item_price < PRICE_OUT) & (gb_train.item_cnt_month < SALES_OUT)]
gb_train = gb_train.pivot_table(index=['shop_id', 'item_id'], values=['item_cnt_month'], columns=['date_block_num'], fill_value=0)
gb_train.head(1)
df_all_data = pd.merge(df_test, gb_train, on=['item_id', 'shop_id'], how='left')
df_all_data.fillna(0, inplace=True)
df_all_data.drop(['ID', 'shop_id', 'item_id'], inplace=True, axis=1)
df_all_data.head(1)
X_train = np.expand_dims(df_all_data.values[:, :-1], axis=2)
y_train = df_all_data.values[:, -1:]
X_test = np.expand_dims(df_all_data.values[:, 1:], axis=2)
print(X_train.shape, y_train.shape, X_test.shape)
model_one = Sequential()
model_one.add(LSTM(units=64, input_shape=(33, 1)))
model_one.add(Dropout(0.4))
model_one.add(Dense(1))
model_one.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])
model_one.summary()