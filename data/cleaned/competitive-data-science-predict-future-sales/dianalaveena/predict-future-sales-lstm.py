import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from collections import Counter
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import datetime as dt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dropout, Dense, BatchNormalization
from tensorflow.keras.models import Sequential
import warnings
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
train
(train.shape, test.shape)
train.dtypes
train.describe().style.background_gradient()
test.describe().style.background_gradient()
plt.figure(figsize=(20, 5))
ax = sbn.countplot(data=train, x='shop_id')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_title('Frequency of Shop ID in train data')

plt.figure(figsize=(20, 5))
ax = sbn.countplot(data=test, x='shop_id')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_title('Frequency of Shop ID in test data')

counter = pd.DataFrame(Counter(train[train['item_cnt_day'] < 0]['item_id']).most_common(10))
counter.columns = ['item_id', 'Counts']
counter['item_id'] = counter['item_id'].astype('str')
plt.figure(figsize=(20, 5))
plt.barh(data=counter, y='item_id', width='Counts', color='blue')
plt.title('10 most Items returned by the customers')

plt.figure(figsize=(20, 5))
ax = sbn.countplot(data=train[train['item_cnt_day'] < 0], x='shop_id')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_title('Items returned to the shop by customers')

len(train.item_id.unique())
len(test.item_id.unique())
columns = set(test['item_id']) - set(train['item_id'])
columns
test_data = test.copy(deep=True)
train = train.drop(['date_block_num', 'item_price'], axis=1)
test = test.drop(['ID'], axis=1)
(train.shape, test.shape)
train.isna().sum()
test.isna().sum()

def get_month(x):
    return dt.datetime(x.year, x.month, 1)
train['date'] = [x.replace('.', '-') for x in train['date']]
train['date'] = pd.to_datetime(train['date'], format='%d-%m-%Y')
train['date'] = train['date'].apply(get_month)
train.head()
train = train.groupby(['date', 'shop_id', 'item_id']).agg({'item_cnt_day': 'sum'})
train.reset_index(inplace=True)
train['shop_item'] = train['shop_id'].astype('str') + '_' + train['item_id'].astype('str')
test['shop_item'] = test['shop_id'].astype('str') + '_' + test['item_id'].astype('str')
train = train.drop(['shop_id', 'item_id'], axis=1)
test = test.drop(['shop_id', 'item_id'], axis=1)
train.head()
data = train.merge(test, how='outer', on='shop_item').fillna(0)
warnings.filterwarnings('ignore')
data.date[data.date == 0] = pd.to_datetime('2015-10-01')
data
data = data.pivot_table(index='date', columns='shop_item').item_cnt_day.fillna(0)
data
data = data.loc[:, test['shop_item']]
data
n_past = 1
n_future = 1
trainX = []
trainY = []
testX = []
for i in range(n_past, len(np.array(data)) - n_future + 1):
    trainY.append(np.array(data)[i + n_future - 1:i + n_future])
    trainX.append(np.array(data)[i - n_past:i, 0:np.array(data).shape[1]])
(trainX, trainY) = (np.array(trainX), np.array(trainY))
(trainX.shape, trainY.shape)
model = Sequential()
model.add(LSTM(16, activation='tanh', input_shape=(trainX.shape[1], trainX.shape[2]), batch_input_shape=(1, trainX.shape[1], trainX.shape[2]), return_sequences=True, stateful=True))
model.add(LSTM(16, activation='tanh', return_sequences=False, stateful=False))
model.add(Dense(16, activation='tanh'))
model.add(Dense(trainY.shape[2], activation='linear'))
model.summary()
model.compile(loss='huber_loss', optimizer=Adam(learning_rate=5e-05, clipnorm=1.0), metrics='mse')