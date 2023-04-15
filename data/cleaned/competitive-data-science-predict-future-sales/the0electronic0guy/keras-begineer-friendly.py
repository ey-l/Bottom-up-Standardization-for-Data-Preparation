import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
sample = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
df.head()
sample.head()
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
items_category = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
items.head()
items.shape
items_category.shape
items.head(5)
items = items.drop(columns=['item_name', 'item_name'])
items_category.head()
shops.head()
shops.shape
category = []
for i in df['item_id']:
    category.append(items['item_category_id'][i])
print(category[0:20])
items.iloc[22154, :]
df['item_category_id'] = category
df.head()
data = df[df['item_cnt_day'] <= 50]
data = data[data['item_cnt_day'] > -3]
data.shape
df.shape
len(data['item_cnt_day'].unique())
data_train = data.drop(columns=['item_cnt_day', 'item_id', 'date'])
target = data['item_cnt_day']
print(target.value_counts())
data.shape
data.head()
data.drop(columns=['date', 'item_id'], inplace=True)
date_block = []
for i in data['date_block_num']:
    date_block.append(i % 12)
data['date_block_engineered'] = date_block
data['date_block_engineered'].unique()
data.drop(columns=['date_block_num'], inplace=True)
data.head()
import tensorflow.keras as keras
labels = data['item_cnt_day']
len(labels.unique())
labels.head()
labels.shape
data.head()
data = data.drop(columns=['item_cnt_day'])
data.shape
labels.shape
label = labels.transpose()
keras.backend.clear_session()
model = keras.models.Sequential([keras.layers.Dense(3, input_dim=2, activation='relu'), keras.layers.Dense(1, activation='relu')])
early = keras.callbacks.EarlyStopping(patience=5)
model_check = keras.callbacks.ModelCheckpoint('model.h5', save_best_only=True)
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
data.head()
data_x = data.drop(columns=['date_block_engineered', 'item_price'])
data_x.head()