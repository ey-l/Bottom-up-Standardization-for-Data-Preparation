import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
items_categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
sales_train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
sample_submission = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
test.head()
(items.columns, items_categories.columns, sales_train.columns, shops.columns)
fiyat = sales_train[sales_train['item_price'] > 10000]
fiyat.drop(['shop_id', 'date', 'date_block_num', 'item_cnt_day'], inplace=True, axis=1)
fiyat
print('Dataset büyüklüğü:', sales_train.shape)
satış = sales_train[sales_train['item_id'].isin(test['item_id'].unique())]
print('Aynı idli ürünleri çıkarınca dataset büyüklüğü:', satış.shape)
sns.set_context('talk', font_scale=0.8)
mağaza_satış = pd.DataFrame(sales_train.groupby(['shop_id']).sum().item_cnt_day).reset_index()
mağaza_satış.columns = ['shop_id', 'sum_sales']
sns.barplot(x='shop_id', y='sum_sales', data=mağaza_satış, palette='Paired')
del mağaza_satış
from datetime import datetime
sales_train['year'] = pd.to_datetime(sales_train['date']).dt.strftime('%Y')
sales_train['month'] = sales_train.date.apply(lambda x: datetime.strptime(x, '%d.%m.%Y').strftime('%m'))

grafik = pd.DataFrame(sales_train.groupby(['year', 'month'])['item_cnt_day'].sum().reset_index())
sns.pointplot(x='month', y='item_cnt_day', hue='year', data=grafik)
pivot = sales_train.copy()
pivot = pivot.pivot_table(index=['item_id', 'shop_id'], values='item_cnt_day', columns='date_block_num', fill_value=0, aggfunc='sum').reset_index()
pivot.head(10)
pivot = pd.merge(test, pivot, on=['shop_id', 'item_id'], how='left')
pivot.head(10)
pivot.fillna(0, inplace=True)
pivot.head(10)
X_train = np.expand_dims(pivot.values[:, :-1], axis=2)
y_train = pivot.values[:, -1:]
X_test = np.expand_dims(pivot.values[:, 1:], axis=2)
print(X_train.shape, y_train.shape, X_test.shape)
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.models import load_model, Model
satis_modeli = Sequential()
satis_modeli.add(LSTM(units=64, input_shape=(33, 1)))
satis_modeli.add(Dropout(0.5))
satis_modeli.add(Dense(1))
satis_modeli.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])
satis_modeli.summary()