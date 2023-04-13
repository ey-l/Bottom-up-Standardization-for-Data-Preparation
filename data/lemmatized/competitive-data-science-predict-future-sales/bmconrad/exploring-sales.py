import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
_input4 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
_input3 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
_input1 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
_input0 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
_input2 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
_input5 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
print(_input0.head())
print(_input2.head())
print(_input5.head())
print(_input5.tail())
import datetime
_input0['date'] = pd.to_datetime(_input0['date'], format='%d.%m.%Y')
_input0['year'] = _input0.date.apply(lambda x: x.year)
_input0['month'] = _input0.date.apply(lambda x: x.month)
_input0.date.head()
df_agg = pd.DataFrame(_input0.groupby(by=['month', 'year']).item_cnt_day.sum().reset_index())
sns.pointplot(x='month', y='item_cnt_day', hue='year', data=df_agg)
x = _input0.groupby(by=['date_block_num']).item_cnt_day.sum()
plt.plot(x)
print('Data set size before remove item price 0 cleaning:', _input0.shape)
sales = _input0.query('item_price > 0')
print('Data set size after remove item price 0 cleaning:', sales.shape)
print('Data set size before filter valid:', sales.shape)
sales = sales[sales['shop_id'].isin(_input2['shop_id'].unique())]
sales = sales[sales['item_id'].isin(_input2['item_id'].unique())]
print('Data set size after filter valid:', sales.shape)
print('Data set size before remove outliers:', sales.shape)
sales = sales.query('item_cnt_day >= 0 and item_cnt_day <= 125 and item_price < 75000')
print('Data set size after remove outliers:', sales.shape)
monthly_sales = _input0.groupby(by=['date_block_num', 'item_id', 'shop_id']).agg({'date': ['min', 'max'], 'item_cnt_day': 'sum', 'item_price': 'mean'})
monthly_sales.head(50)
_input2.head()
sales_data_flat = monthly_sales.item_cnt_day.reset_index()
sales_data_flat.head()
sales_data_flat = pd.merge(_input2, sales_data_flat, on=['item_id', 'shop_id'], how='left')
sales_data_flat.head()
sales_data_flat = sales_data_flat.fillna(0, inplace=False)
sales_data_flat = sales_data_flat.drop(['shop_id', 'item_id'], inplace=False, axis=1)
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