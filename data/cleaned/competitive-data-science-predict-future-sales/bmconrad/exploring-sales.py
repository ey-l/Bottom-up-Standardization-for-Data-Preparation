import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
items_categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
sales_train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
sales_test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
example = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
print(sales_train.head())
print(sales_test.head())
print(example.head())
print(example.tail())
import datetime
sales_train['date'] = pd.to_datetime(sales_train['date'], format='%d.%m.%Y')
sales_train['year'] = sales_train.date.apply(lambda x: x.year)
sales_train['month'] = sales_train.date.apply(lambda x: x.month)
sales_train.date.head()
df_agg = pd.DataFrame(sales_train.groupby(by=['month', 'year']).item_cnt_day.sum().reset_index())
sns.pointplot(x='month', y='item_cnt_day', hue='year', data=df_agg)
x = sales_train.groupby(by=['date_block_num']).item_cnt_day.sum()
plt.plot(x)
print('Data set size before remove item price 0 cleaning:', sales_train.shape)
sales = sales_train.query('item_price > 0')
print('Data set size after remove item price 0 cleaning:', sales.shape)
print('Data set size before filter valid:', sales.shape)
sales = sales[sales['shop_id'].isin(sales_test['shop_id'].unique())]
sales = sales[sales['item_id'].isin(sales_test['item_id'].unique())]
print('Data set size after filter valid:', sales.shape)
print('Data set size before remove outliers:', sales.shape)
sales = sales.query('item_cnt_day >= 0 and item_cnt_day <= 125 and item_price < 75000')
print('Data set size after remove outliers:', sales.shape)
monthly_sales = sales_train.groupby(by=['date_block_num', 'item_id', 'shop_id']).agg({'date': ['min', 'max'], 'item_cnt_day': 'sum', 'item_price': 'mean'})
monthly_sales.head(50)
sales_test.head()
sales_data_flat = monthly_sales.item_cnt_day.reset_index()
sales_data_flat.head()
sales_data_flat = pd.merge(sales_test, sales_data_flat, on=['item_id', 'shop_id'], how='left')
sales_data_flat.head()
sales_data_flat.fillna(0, inplace=True)
sales_data_flat.drop(['shop_id', 'item_id'], inplace=True, axis=1)
sales_data_flat.head()
pivoted_sales = sales_data_flat.pivot_table(index='ID', columns='date_block_num', fill_value=0, aggfunc='sum')
pivoted_sales.head(20)
pivoted_sales = sales_data_flat.pivot_table(index='ID', columns='date_block_num', fill_value=0, aggfunc='sum')
pivoted_sales.head()
X_train = np.expand_dims(pivoted_sales.values[:, :-1], axis=2)
y_train = pivoted_sales.values[:, -1:]
X_test = np.expand_dims(pivoted_sales.values[:, 1:], axis=2)
print(X_train.shape, y_train.shape, X_test.shape)
X_train = np.expand_dims(pivoted_sales.values[:, :-1], axis=2)
y_train = pivoted_sales.values[:, -1:]
X_test = np.expand_dims(pivoted_sales.values[:, 1:], axis=2)
print('X_train.shape {} y_train.shape {} X_test.shape {}'.format(X_train.shape, y_train.shape, X_test.shape))
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.models import load_model, Model
sales_model = Sequential()
sales_model.add(LSTM(units=64, input_shape=(33, 1)))
sales_model.add(Dropout(0.5))
sales_model.add(Dense(1))
sales_model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])
sales_model.summary()