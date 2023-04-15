import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport as pp
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
import warnings
warnings.filterwarnings('ignore')
sales = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
item_cat = pd.read_csv('_data/input/competitive-data-science-predict-future-sales//item_categories.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales//items.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales//shops.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales//test.csv')
submission = pd.read_csv('_data/input/competitive-data-science-predict-future-sales//sample_submission.csv')
sales.dtypes
sales.head(5)
from datetime import datetime
sales = sales[(sales['item_cnt_day'] > 0) & (sales['item_price'] > 0)]
sales['date'] = pd.to_datetime(sales['date'])
sales.describe()
sales.shape
item_cat.info()
item_cat.head()
item_cat.shape
test.info()
test.head()
test.shape
submission.head()
sales_group = sales.groupby(['date_block_num', 'shop_id', 'item_id'])['item_cnt_day'].sum()
sales_group.head(20)
sales_month_group = sales.groupby('date_block_num')['item_cnt_day'].sum()
sales_month_group.head()
plt.figure(figsize=(16, 8))
plt.plot(sales_month_group)
plt.title('month_item_cnt')
plt.xlabel('month')
plt.ylabel('item_cnt_sum')

sales_stack = sales.pivot_table(index=['shop_id', 'item_id'], values=['item_cnt_day'], columns=['date_block_num'], fill_value=0, aggfunc='sum').reset_index()
sales_stack
sales_stack.shape
sales_month_full = pd.merge(test, sales_stack, on=['shop_id', 'item_id'], how='left')
sales_month_full.head()
sales_month_full.fillna(0, inplace=True)
sales_month_full.head()
sales_month_full = sales_month_full.drop(['ID', 'shop_id', 'item_id'], axis=1)
sales_month_full
sales_month_full.shape
(shop_item_train, sales_cnt_train) = (sales_month_full.values[:, :-2], sales_month_full.values[:, -2:-1].ravel())
(shop_item_valid, sales_cnt_valid) = (sales_month_full.values[:, 1:-1], sales_month_full.values[:, -1:].ravel())
shop_item_test = sales_month_full.values[:, 2:]
shop_item_train
sales_cnt_train
shop_item_valid
sales_cnt_valid
shop_item_train.shape
shop_item_test
import math
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
lstm_model = tf.keras.Sequential([tf.keras.layers.Reshape(input_shape=(32,), target_shape=(32, 1)), tf.keras.layers.LSTM(units=32, input_shape=(32, 1)), tf.keras.layers.Dropout(0.4), tf.keras.layers.Dense(1)])
lstm_model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])
lstm_model.summary()