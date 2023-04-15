import pandas as pd
fname_sales_train = '_data/input/competitive-data-science-predict-future-sales/sales_train.csv'
df_sales_train = pd.read_csv(fname_sales_train)
df_sales_train
df_sales_train.isnull().sum()
fname_shops = '_data/input/competitive-data-science-predict-future-sales/shops.csv'
df_shops = pd.read_csv(fname_shops)
df_shops
df_shops.isnull().sum()
fname_items = '_data/input/competitive-data-science-predict-future-sales/items.csv'
df_items = pd.read_csv(fname_items)
df_items
df_items.isnull().sum()
fname_item_categories = '_data/input/competitive-data-science-predict-future-sales/item_categories.csv'
df_item_categories = pd.read_csv(fname_item_categories)
df_item_categories
df_item_categories.isnull().sum()
fname_test = '_data/input/competitive-data-science-predict-future-sales/test.csv'
df_test = pd.read_csv(fname_test)
df_test
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid')
plt.figure(figsize=(12, 3))
sns.boxplot(data=df_sales_train.item_price, orient='h')
df_sales_train = df_sales_train[df_sales_train.item_price < 100000]
plt.figure(figsize=(12, 3))
sns.boxplot(data=df_sales_train.item_cnt_day, orient='h')
df_sales_train = df_sales_train[df_sales_train.item_cnt_day < 900]
df = df_sales_train.pivot_table(index=['shop_id', 'item_id'], columns='date_block_num', values='item_cnt_day', fill_value=0, aggfunc='sum')
df
df = df.reset_index()
df.columns.name = None
df
df_test_src = pd.merge(df_test, df, how='left', on=['shop_id', 'item_id'])
df_test_src = df_test_src.drop(['ID'], axis=1).fillna(0)
df_test_src
X_train = df.iloc[:, 2:-1]
X_train
y_train = df.iloc[:, -1]
y_train
X_test = df_test_src.iloc[:, 3:]
X_test
from sklearn.preprocessing import MinMaxScaler
X_scaler = MinMaxScaler()
X_train = X_scaler.fit_transform(X_train)
X_test = X_scaler.transform(X_test)
import tensorflow as tf
input = tf.keras.layers.Input(shape=(33, 1))
x = input
x = tf.keras.layers.LSTM(32, return_sequences=True, dropout=0.1)(x)
x = tf.keras.layers.LSTM(32, return_sequences=True, dropout=0.1)(x)
x = tf.keras.layers.LSTM(32)(x)
output = tf.keras.layers.Dense(1)(x)
model = tf.keras.models.Model(input, output)
model.summary()
model.compile(optimizer='adam', loss='mse')
import numpy as np
epochs = 100
batch_size = 64
callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]