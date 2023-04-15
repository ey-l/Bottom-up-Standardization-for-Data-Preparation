import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
import numpy as np
import pandas as pd
import copy
from pathlib import Path
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras import optimizers, Sequential, Model
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
item_cat = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
ss = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
train.head()
items.head()
ss.head()
test.tail()
print(test.ID.nunique())
print(test.shape)
test.info()
print(train.item_id.nunique())
print(train.shop_id.nunique())
train.head()
train.shape
import seaborn as sns
sns.heatmap(train.isnull())
print(test[~test.item_id.isin(train.item_id)].shape)
print(test[~test.shop_id.isin(train.shop_id)].shape)
print(test.ID.nunique())
print(test.shop_id.nunique())
print(test.item_id.nunique())
test.head()
print(items.item_id.nunique())
print(items.item_category_id.nunique())
print(test[~test.item_id.isin(items.item_id)].shape)
items.head()
print(train[~train.item_id.isin(items.item_id)]['item_id'].nunique())
print(test[~test.item_id.isin(train.item_id)]['item_id'].nunique())
train = train[train['shop_id'].isin(test['shop_id'])]
train = train[train['item_id'].isin(test['item_id'].unique())]
train_data = pd.merge(train, items, on='item_id', how='inner')
train_data.drop('item_name', axis=1, inplace=True)
train_data.head()
train_data['item_cnt_month'] = train_data.groupby(['shop_id', 'item_id', 'date_block_num'])['item_cnt_day'].transform('sum')
train_data['monthly_sales'] = train_data.groupby('date_block_num')['item_cnt_day'].transform('sum')
train_data.head()
sns.lineplot(x='date_block_num', y='monthly_sales', data=train_data)
print(train_data['item_cnt_month'].min())
print(train_data['item_cnt_month'].max())
print(train_data['item_cnt_month'].mean())
print(train_data['item_cnt_month'].median())
train_data = train_data[(train_data.item_cnt_month >= 0) & (train_data.item_cnt_month <= 15)]
print(train_data['item_cnt_month'].min())
print(train_data['item_cnt_month'].max())
print(train_data['item_cnt_month'].mean())
print(train_data['item_cnt_month'].median())
mat = train_data.pivot_table(index=['shop_id', 'item_id'], columns='date_block_num', values='item_cnt_month', fill_value=0).reset_index()
mat.head()
first = 20
last = 33
sub_series = 12
l = []
for (index, row) in mat.iterrows():
    for i in range(last - (first + sub_series) + 1):
        x = [row['shop_id'], row['item_id']]
        for j in range(sub_series + 1):
            x.append(row[i + first + j])
        l.append(x)
columns = ['shop_id', 'item_id']
[columns.append(i) for i in range(sub_series)]
columns.append('label')
mat1 = pd.DataFrame(l, columns=columns)
mat1.head()
mat1[(mat1['shop_id'] == 2) & (mat1['item_id'] == 31)]
y = mat1['label']
mat1.drop(['label', 'shop_id', 'item_id'], axis=1, inplace=True)
(X_train, X_valid, y_train, y_valid) = train_test_split(mat1, y.values, test_size=0.1, random_state=0)
print(X_train.shape)
print(X_valid.shape)
X_train1 = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
X_valid1 = X_valid.values.reshape((X_valid.shape[0], X_valid.shape[1], 1))
lstm_model = Sequential()
lstm_model.add(LSTM(X_train1.shape[1], input_shape=(X_train1.shape[1], X_train1.shape[2]), return_sequences=True))
lstm_model.add(LSTM(6, activation='relu', return_sequences=True))
lstm_model.add(LSTM(1, activation='relu'))
lstm_model.add(Dense(10, kernel_initializer='glorot_normal', activation='relu'))
lstm_model.add(Dense(10, kernel_initializer='glorot_normal', activation='relu'))
lstm_model.add(Dense(1))
lstm_model.summary()
adam = optimizers.Adam(0.0001)
lstm_model.compile(loss='mse', optimizer=adam)