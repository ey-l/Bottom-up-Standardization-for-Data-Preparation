import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
train.head(2)
item = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
item.head(2)
cat = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
cat.head(2)
shop = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
shop.head(2)
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
test.head(2)
train.head()
test.shape
submission = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
submission.head(2)
submission.shape
train.shape
train = train[train.item_id.isin(test.item_id)]
train = train[train.shop_id.isin(test.shop_id)]
train.info()
train.head()
train.drop(['date'], axis=1, inplace=True)
test.head()
train['date_block_num']
test['date_block_num'] = 34
test = test[['date_block_num', 'shop_id', 'item_id']]
test.head(2)
item_price = dict(train.groupby('item_id')['item_price'].last().reset_index().values)
test['item_price'] = test.item_id.map(item_price)
test.head()
test.isnull().sum()
(train.shape, test.shape)
train = train[train.item_id.isin(test.item_id)]
train = train[train.shop_id.isin(test.shop_id)]
(train.shape, test.shape)
test.isnull().sum()
train['shop*item'] = train.shop_id * train.item_id
test['shop*item'] = test.shop_id * test.item_id
item.head()
item.drop('item_name', axis=1, inplace=True)
item_cat = dict(item.values)
train['item_cat'] = train.item_id.map(item_cat)
test['item_cat'] = test.item_id.map(item_cat)
train.head(2)
train.info()
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.concat([train, test])
sns.histplot(df['item_price'])
df = pd.concat([train, test])
df.item_price = np.log1p(df.item_price)
df.item_price = df.item_price.fillna(df.item_price.mean())
df.item_cnt_day = df.item_cnt_day.apply(lambda x: 10 if x > 10 else x)
train = df[df.item_cnt_day.notnull()]
test = df[df.item_cnt_day.isnull()]
train.shape
test.isnull().sum()
test.drop('item_cnt_day', axis=1, inplace=True)
test.shape
x_train = train.drop('item_cnt_day', axis=1).values
y_train = train.item_cnt_day.values
x_test = test
from sklearn.preprocessing import MinMaxScaler
SC = MinMaxScaler()
x_train = SC.fit_transform(x_train)
x_test = SC.transform(x_test)
import keras
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(9, kernel_initializer='uniform', activation='relu', input_dim=6))
model.add(Dense(9, kernel_initializer='uniform', activation='relu'))
model.add(Dense(5, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='linear'))
model.summary()
model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mse', 'mae'])