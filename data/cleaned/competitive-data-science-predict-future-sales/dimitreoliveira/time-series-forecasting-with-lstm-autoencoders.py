import os, warnings, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras import optimizers, Sequential, Model

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
seed = 0
seed_everything(seed)
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: '%.2f' % x)
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv', dtype={'ID': 'int32', 'shop_id': 'int32', 'item_id': 'int32'})
item_categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv', dtype={'item_category_name': 'str', 'item_category_id': 'int32'})
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv', dtype={'item_name': 'str', 'item_id': 'int32', 'item_category_id': 'int32'})
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv', dtype={'shop_name': 'str', 'shop_id': 'int32'})
sales = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv', parse_dates=['date'], dtype={'date': 'str', 'date_block_num': 'int32', 'shop_id': 'int32', 'item_id': 'int32', 'item_price': 'float32', 'item_cnt_day': 'int32'})
train = sales.join(items, on='item_id', rsuffix='_').join(shops, on='shop_id', rsuffix='_').join(item_categories, on='item_category_id', rsuffix='_').drop(['item_id_', 'shop_id_', 'item_category_id_'], axis=1)
print(f'Train rows: {train.shape[0]}')
print(f'Train columns: {train.shape[1]}')


print(f"Min date from train set: {train['date'].min().date()}")
print(f"Max date from train set: {train['date'].max().date()}")
test_shop_ids = test['shop_id'].unique()
test_item_ids = test['item_id'].unique()
train = train[train['shop_id'].isin(test_shop_ids)]
train = train[train['item_id'].isin(test_item_ids)]
train_monthly = train[['date', 'date_block_num', 'shop_id', 'item_id', 'item_cnt_day']]
train_monthly = train_monthly.sort_values('date').groupby(['date_block_num', 'shop_id', 'item_id'], as_index=False)
train_monthly = train_monthly.agg({'item_cnt_day': ['sum']})
train_monthly.columns = ['date_block_num', 'shop_id', 'item_id', 'item_cnt']
train_monthly = train_monthly.query('item_cnt >= 0 and item_cnt <= 20')
train_monthly['item_cnt_month'] = train_monthly.sort_values('date_block_num').groupby(['shop_id', 'item_id'])['item_cnt'].shift(-1)


monthly_series = train_monthly.pivot_table(index=['shop_id', 'item_id'], columns='date_block_num', values='item_cnt', fill_value=0).reset_index()
monthly_series.head()
first_month = 20
last_month = 33
serie_size = 12
data_series = []
for (index, row) in monthly_series.iterrows():
    for month1 in range(last_month - (first_month + serie_size) + 1):
        serie = [row['shop_id'], row['item_id']]
        for month2 in range(serie_size + 1):
            serie.append(row[month1 + first_month + month2])
        data_series.append(serie)
columns = ['shop_id', 'item_id']
[columns.append(i) for i in range(serie_size)]
columns.append('label')
data_series = pd.DataFrame(data_series, columns=columns)
data_series.head()
data_series = data_series.drop(['item_id', 'shop_id'], axis=1)
labels = data_series['label']
data_series.drop('label', axis=1, inplace=True)
(train, valid, Y_train, Y_valid) = train_test_split(data_series, labels.values, test_size=0.1, random_state=0)
print('Train set', train.shape)
print('Validation set', valid.shape)
train.head()
X_train = train.values.reshape((train.shape[0], train.shape[1], 1))
X_valid = valid.values.reshape((valid.shape[0], valid.shape[1], 1))
print('Train set reshaped', X_train.shape)
print('Validation set reshaped', X_valid.shape)
serie_size = X_train.shape[1]
n_features = X_train.shape[2]
epochs = 20
batch = 128
lr = 0.0001
lstm_model = Sequential()
lstm_model.add(L.LSTM(10, input_shape=(serie_size, n_features), return_sequences=True))
lstm_model.add(L.LSTM(6, activation='relu', return_sequences=True))
lstm_model.add(L.LSTM(1, activation='relu'))
lstm_model.add(L.Dense(10, kernel_initializer='glorot_normal', activation='relu'))
lstm_model.add(L.Dense(10, kernel_initializer='glorot_normal', activation='relu'))
lstm_model.add(L.Dense(1))
lstm_model.summary()
adam = optimizers.Adam(lr)
lstm_model.compile(loss='mse', optimizer=adam)