import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from datetime import datetime, date
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
sample_sub = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
item_cats = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
print('----------Top-10- Record----------')
print(train.head(10))
print('-----------Information-----------')
print(train.info())
print('-----------Data Types-----------')
print(train.dtypes)
print('----------Missing value-----------')
print(train.isnull().sum())
print('----------Null value-----------')
print(train.isna().sum())
print('----------Shape of Data----------')
print(train.shape)
train['date'] = pd.to_datetime(train['date'], format='%d.%m.%Y')
sum(train['item_cnt_day'] == 0)
train.head()
dataset = train.pivot_table(index=['shop_id', 'item_id'], values=['item_cnt_day'], columns=['date_block_num'], fill_value=0, aggfunc=np.sum)
dataset.reset_index(inplace=True)
dataset.head()
dataset_left = pd.merge(test, dataset, on=['item_id', 'shop_id'], how='left')
dataset_left.fillna(0, inplace=True)
dataset_left.head()
dataset_left.drop(['shop_id', 'item_id', 'ID'], inplace=True, axis=1)
dataset_left.head()
X_train = np.expand_dims(dataset_left.values[:, :-1], axis=2)
y_train = dataset_left.values[:, -1:]
X_test = np.expand_dims(dataset_left.values[:, 1:], axis=2)
print(X_train.shape, y_train.shape, X_test.shape)
model = keras.models.Sequential([keras.layers.LSTM(64, return_sequences=True, input_shape=[33, 1]), keras.layers.LSTM(64), keras.layers.Dropout(0.4), keras.layers.Dense(1)])
model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])