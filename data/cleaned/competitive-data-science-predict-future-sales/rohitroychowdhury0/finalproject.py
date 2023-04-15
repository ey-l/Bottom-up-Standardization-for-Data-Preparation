import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import os
import matplotlib.pyplot as plt

sales_data = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
item_cat = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
sample_submission = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
test_data = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
sales_data.head(10)
sample_submission.head()
sales_data['date'] = pd.to_datetime(sales_data['date'], format='%d.%m.%Y')
sales_data.head()
dataset = sales_data.pivot_table(index=['shop_id', 'item_id'], values=['item_cnt_day'], columns=['date_block_num'], fill_value=0, aggfunc='sum')
dataset.head()
dataset.reset_index(inplace=True)
dataset.head(10)
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