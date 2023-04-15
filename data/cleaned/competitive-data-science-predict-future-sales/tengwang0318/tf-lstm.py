import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sale_data = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv', parse_dates=['date'])
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
item_cat = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')

def EDA(df):
    print('-------df.head(5)-------')
    print(df.head(5))
    print('-------INFO----------')
    print(df.info())
    print('-------describe--------')
    print(df.describe())
    print('-------data type-------')
    print(df.dtypes)
    print('-------missing num-------')
    print(df.isna().sum())
EDA(sale_data)
dataset = sale_data.pivot_table(index=['shop_id', 'item_id'], values=['item_cnt_day'], columns=['date_block_num'], fill_value=0, aggfunc='sum')
dataset
dataset.reset_index(inplace=True)
dataset
dataset = pd.merge(test, dataset, on=['item_id', 'shop_id'], how='left')
dataset
dataset.fillna(0, inplace=True)
dataset
dataset.drop(['shop_id', 'item_id', 'ID'], inplace=True, axis=1)
dataset
x_train = np.expand_dims(dataset.values[:, :-1], axis=2)
y = dataset.values[:, -1:]
x_test = np.expand_dims(dataset.values[:, 1:], axis=2)
print(x_train.shape, y.shape, x_test.shape)
model = tf.keras.Sequential([tf.keras.layers.LSTM(64, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True), tf.keras.layers.LSTM(128, return_sequences=True), tf.keras.layers.LSTM(32), tf.keras.layers.Dropout(0.3), tf.keras.layers.Dense(1, activation='relu')])
model.summary()
model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])