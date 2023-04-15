import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
item_categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
sales = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv', parse_dates=['date'], infer_datetime_format=True, dayfirst=True)
sales2 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv', parse_dates=['date'], infer_datetime_format=True, dayfirst=True)
df_item = pd.merge(items, item_categories, on='item_category_id', how='inner')
sales_t = pd.merge(sales, shops, on='shop_id', how='inner')
sales = pd.merge(sales_t, df_item, on='item_id', how='inner')
df_item2 = pd.merge(items, item_categories, on='item_category_id', how='inner')
sales_t2 = pd.merge(sales2, shops, on='shop_id', how='inner')
sales2 = pd.merge(sales_t2, df_item2, on='item_id', how='inner')
sales = sales[sales['shop_id'].isin(test['shop_id'].unique())]
sales = sales[sales['item_id'].isin(test['item_id'].unique())]
sales = sales[(sales.item_price < 300000) & (sales.item_cnt_day < 1000)]
sales = sales[sales.item_price > 0].reset_index(drop=True)
sales2 = sales2[(sales2.item_price < 300000) & (sales2.item_cnt_day < 1000)]
sales2 = sales2[sales2.item_price > 0].reset_index(drop=True)
sales_recien_cargado = sales
sales = sales.groupby(['date_block_num', 'shop_id', 'item_id'])[['date_block_num', 'date', 'shop_id', 'item_id', 'item_price', 'item_cnt_day']].agg({'date_block_num': 'mean', 'date': ['min', 'max'], 'item_price': 'mean', 'item_cnt_day': 'sum'})
sales = sales.item_cnt_day.apply(list).reset_index()
sales_data = pd.merge(test, sales, on=['item_id', 'shop_id'], how='left')
sales2 = sales2.groupby(['date_block_num', 'shop_id', 'item_id'])[['date_block_num', 'date', 'shop_id', 'item_id', 'item_price', 'item_cnt_day']].agg({'date_block_num': 'mean', 'date': ['min', 'max'], 'item_price': 'mean', 'item_cnt_day': 'sum'})
sales2 = sales2.item_cnt_day.apply(list).reset_index()
sales_data2 = sales2
sales_prueba = sales_data
sales_data.fillna(0, inplace=True)
sales_data.drop(['shop_id', 'item_id'], inplace=True, axis=1)
sales_data = sales_data.pivot_table(index='ID', columns='date_block_num', values='sum', aggfunc='sum')
sales_data = sales_data.fillna(0)
sales_data2.fillna(0, inplace=True)
sales_data2 = sales_data2.pivot_table(index=['shop_id', 'item_id'], columns='date_block_num', values='sum', aggfunc='sum')
sales_data2 = sales_data2.fillna(0)
x_train = sales_data2[sales_data2.columns[:-1]]
y_train = sales_data2[sales_data2.columns[-1]]
x_test = sales_data[sales_data.columns[:-1]]
y_test = sales_data[sales_data.columns[-1]]
model = LinearRegression()