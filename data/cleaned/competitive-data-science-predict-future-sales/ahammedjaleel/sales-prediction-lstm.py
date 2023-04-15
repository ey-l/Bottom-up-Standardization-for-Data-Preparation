import os
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style='darkgrid')
pd.set_option('display.float_format', lambda x: '%.2f' % x)
sales = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
item_cats = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
print('sales:', sales.shape, 'test:', test.shape, 'items:', items.shape, 'item_cats:', item_cats.shape, 'shop:', shops.shape)
sales['date'] = pd.to_datetime(sales['date'], format='%d.%m.%Y')
sales.head(3)
test.head(3)
test.shape
items.head(3)
item_cats.head(3)
shops.head(3)
sales[sales['item_price'] <= 0]
sales[(sales.shop_id == 32) & (sales.item_id == 2973) & (sales.date_block_num == 4)]
median = sales[(sales.shop_id == 32) & (sales.item_id == 2973) & (sales.date_block_num == 4) & (sales.item_price > 0)].item_price.median()
sales.loc[sales.item_price < 0, 'item_price'] = median
dataset = sales.pivot_table(index=['shop_id', 'item_id'], values=['item_cnt_day'], columns=['date_block_num'], fill_value=0, aggfunc='sum')
dataset
dataset.reset_index(inplace=True)
dataset.head()
dataset.shape
dataset = pd.merge(test, dataset, on=['item_id', 'shop_id'], how='left')
dataset
dataset.fillna(0, inplace=True)
dataset.head()
dataset.drop(['shop_id', 'item_id', 'ID'], inplace=True, axis=1)
dataset.head()
X_train = np.expand_dims(dataset.values[:, :-1], axis=2)
y_train = dataset.values[:, -1:]
X_test = np.expand_dims(dataset.values[:, 1:], axis=2)
print(X_train.shape, y_train.shape, X_test.shape)
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
my_model = Sequential()
my_model.add(LSTM(units=64, input_shape=(33, 1)))
my_model.add(Dropout(0.3))
my_model.add(Dense(1))
my_model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])
my_model.summary()