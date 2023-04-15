import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import seaborn as sns
import scipy
sales_data = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
item_cat = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
sample_submission = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
test_data = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
sales_data.head(3)
test_data.head(3)
sample_submission.head(3)

def basic_eda(df):
    print('----------TOP 5 RECORDS--------')
    print(df.head(5))
    print('----------INFO-----------------')
    print(df.info())
    print('----------Describe-------------')
    print(df.describe())
    print('----------Columns--------------')
    print(df.columns)
    print('----------Data Types-----------')
    print(df.dtypes)
    print('-------Missing Values----------')
    print(df.isnull().sum())
    print('-------NULL values-------------')
    print(df.isna().sum())
    print('-----Shape Of Data-------------')
    print(df.shape)
print('=============================Sales Data=============================')
basic_eda(sales_data)
print('=============================Test data=============================')
basic_eda(test_data)
print('=============================Item Categories=============================')
basic_eda(item_cat)
print('=============================Items=============================')
basic_eda(items)
print('=============================Shops=============================')
basic_eda(shops)
print('=============================Sample Submission=============================')
basic_eda(sample_submission)
sales_data['date'] = pd.to_datetime(sales_data['date'], format='%d.%m.%Y')
sales_data.head(3)
dataset = sales_data.pivot_table(index=['shop_id', 'item_id'], values=['item_cnt_day'], columns=['date_block_num'], fill_value=0, aggfunc='sum')
dataset.head(3)
dataset.reset_index(inplace=True)
dataset.head(3)
dataset = pd.merge(test_data, dataset, on=['item_id', 'shop_id'], how='left')
dataset.head()
dataset.fillna(0, inplace=True)
dataset.head()
dataset.drop(['shop_id', 'item_id', 'ID'], inplace=True, axis=1)
dataset.head()
X_train = np.expand_dims(dataset.values[:, :-1], axis=2)
y_train = dataset.values[:, -1:]
X_test = np.expand_dims(dataset.values[:, 1:], axis=2)
print(X_train.shape, y_train.shape, X_test.shape)
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten, Dropout
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
model_lstm = Sequential()
model_lstm.add(LSTM(units=64, input_shape=(X_train.shape[1], X_train.shape[2])))
model_lstm.add(Dropout(0.4))
model_lstm.add(Dense(1))
model_lstm.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])
model_lstm.summary()