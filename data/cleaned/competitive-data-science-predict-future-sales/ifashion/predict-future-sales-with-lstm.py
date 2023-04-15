import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
default_path = '_data/input/competitive-data-science-predict-future-sales/'

train_df = pd.read_csv(default_path + 'sales_train.csv')
items_df = pd.read_csv(default_path + 'items.csv')
test_df = pd.read_csv(default_path + 'test.csv')
print(train_df.shape, test_df.shape)
train_df.head()
sns.boxplot(y='item_cnt_day', data=train_df)
train_df = train_df.drop_duplicates()
train_df[train_df.duplicated()]
train_df = train_df[train_df['item_cnt_day'] < 1100]
train_df['date'] = pd.to_datetime(train_df['date'], format='%d.%m.%Y')
train_df.info()
train_df.tail()
dataset = train_df.pivot_table(index=['item_id', 'shop_id'], values=['item_cnt_day'], columns='date_block_num', fill_value=0, aggfunc=np.sum)
dataset = dataset.reset_index()
dataset.describe()
dataset.tail()
test_df.head()
dataset = pd.merge(test_df, dataset, on=['item_id', 'shop_id'], how='left')
dataset = dataset.fillna(0)
dataset.head()
dataset = dataset.drop(['shop_id', 'item_id', 'ID'], axis=1)
dataset.head()
X_train = np.expand_dims(dataset.values[:, :-1], axis=2)
y_train = dataset.values[:, -1:]
X_test = np.expand_dims(dataset.values[:, 1:], axis=2)
print(X_train.shape, y_train.shape, X_test.shape)
dataset.values.shape
dataset.values[:, :-1].shape
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
model = Sequential()
model.add(LSTM(units=64, input_shape=(33, 1)))
model.add(Dropout(0.3))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])
model.summary()
from keras.callbacks import EarlyStopping
callbacks_list = [EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3, mode='auto')]