import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
train_ds = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
train_ds.head(10)
test_ds = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
test_ds.head(10)
monthly_data = train_ds.pivot_table(index=['shop_id', 'item_id'], values=['item_cnt_day'], columns=['date_block_num'], fill_value=0, aggfunc='sum')
monthly_data.head(10)
monthly_data.tail(10)
monthly_data.reset_index(inplace=True)
monthly_data.head()
train_data = monthly_data.drop(columns=['shop_id', 'item_id'], level=0)
train_data.head()
train_data.fillna(0, inplace=True)
train_data.head()
x_train = np.expand_dims(train_data.values[:, :-1], axis=2)
y_train = train_data.values[:, -1:]
test_rows = monthly_data.merge(test_ds, on=['item_id', 'shop_id'], how='right')
test_rows.head()
x_test = test_rows.drop(test_rows.columns[:5], axis=1).drop('ID', axis=1)
x_test.fillna(0, inplace=True)
x_test.head()
x_test = np.expand_dims(x_test, axis=2)
print(x_train.shape, y_train.shape, x_test.shape)
model = tf.keras.models.Sequential()
model.add(LSTM(64, input_shape=(33, 1), return_sequences=False))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])