import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
from sklearn.metrics import mean_absolute_error
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
item_categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
item = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')

def basic_info(data):
    print('*************** TOP 10 RECOED ***************')
    print(data.head(10))
    print('*************** INFO ***************')
    print(data.info())
    print('*************** DESCRIBE ***************')
    print(data.describe())
    print('*************** COLUMNS ***************')
    print(data.columns)
    print('*************** MISSING VALUES ***************')
    print(data.isnull().sum())
    print('*************** SHAPE ***************')
    print(data.shape)
print('=============== TRAIN ===============')
basic_info(train)
print('=============== TEST ===============')
basic_info(test)
print('=============== ITEM CAT ===============')
basic_info(item_categories)
print('=============== SHOPS ===============')
basic_info(shops)
print('=============== ITEMS ===============')
basic_info(item)
train['date'] = pd.to_datetime(train['date'], format='%d.%m.%Y')
pt_train = pd.pivot_table(train, index=['shop_id', 'item_id'], values='item_cnt_day', columns=['date_block_num'], aggfunc=np.sum, fill_value=0)
pt_train.reset_index(inplace=True)
pt_test = pd.merge(test, pt_train, on=['shop_id', 'item_id'], how='left')
pt_test.fillna(0, inplace=True)
X = pt_train.drop(columns=['shop_id', 'item_id', 33], axis=1)
y = pt_train[33]
pt_test.drop(columns=['shop_id', 'item_id', 'ID', 0], axis=1, inplace=True)
pt_test.columns = X.columns
X_train = np.expand_dims(X, axis=2)
y_train = np.expand_dims(y, axis=1)
X_test = np.expand_dims(pt_test, axis=2)
model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(LSTM(64, activation='relu', return_sequences=False))
model.add(Dropout(0.1))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])