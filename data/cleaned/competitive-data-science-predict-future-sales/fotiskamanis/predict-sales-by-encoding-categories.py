import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
df_shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
df_items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
df_categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
df_sales = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
df_test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
print(df_shops.shape)
df_shops.head()
print(df_items.shape)
df_items.head()
print(df_categories.shape)
df_categories.head()
print(df_sales.shape)
df_sales.head()
print(df_test.shape)
df_test.head()
agg_item_price = {'item_price': 'item_mean_price'}
df_prices = df_sales.groupby('item_id').agg({'item_price': 'mean'}).rename(columns=agg_item_price)
print(df_prices.shape)
df_prices.head()
df_sales.dtypes
df_sales.isnull().sum()
df_sales.drop(['date_block_num', 'item_price'], axis=1, inplace=True)
print(df_sales.shape)
df_sales.head()
df_sales = pd.merge(df_sales, df_prices, on='item_id', how='left')
print(df_sales.shape)
df_sales.head()
df_sales = df_sales[['date', 'shop_id', 'item_id', 'item_mean_price', 'item_cnt_day']]
print(df_sales.shape)
df_sales.head()
df_sales = pd.merge(df_sales, df_items, on='item_id', how='left')
print(df_sales.shape)
df_sales.head()
df_sales.drop('item_name', axis=1, inplace=True)
print(df_sales.shape)
df_sales.head()
df_sales = df_sales[['date', 'shop_id', 'item_id', 'item_category_id', 'item_mean_price', 'item_cnt_day']]
print(df_sales.shape)
df_sales.head()
df_sales['date'] = pd.to_datetime(df_sales['date'], dayfirst=True)
df_sales['date'] = df_sales['date'].apply(lambda x: x.strftime('%Y-%m'))
print(df_sales.shape)
df_sales.head()
agg_item_cnt = {'item_cnt_day': 'item_sum_qty'}
df_sales = df_sales.groupby(['date', 'shop_id', 'item_id', 'item_category_id', 'item_mean_price']).agg({'item_cnt_day': 'sum'}).rename(columns=agg_item_cnt)
print(df_sales.shape)
df_sales.head()
df_train = df_sales.pivot_table(index=['shop_id', 'item_id', 'item_category_id', 'item_mean_price'], columns='date', values='item_sum_qty', fill_value=0)
df_train.reset_index(inplace=True)
print(df_train.shape)
df_train.head()
df_train_cols1 = ['shop_id', 'item_id', 'item_category_id', 'item_mean_price']
df_train_cols2 = [f'{i}' for i in range(1, 35)]
df_train_cols = df_train_cols1 + df_train_cols2
df_train.columns = df_train_cols
print(df_train.shape)
df_train.head()
X_train = df_train.drop(['34'], axis=1)
Y_train = df_train['34'].values
print(X_train.shape, Y_train.shape)
df_test = pd.merge(df_test, df_train, on=['shop_id', 'item_id'], how='left')
df_test.drop(['ID', '1'], axis=1, inplace=True)
df_test = df_test.fillna(0)
print(df_test.shape)
df_test.head()
df_test_cols1 = ['shop_id', 'item_id', 'item_category_id', 'item_mean_price']
df_test_cols2 = [f'{i}' for i in range(1, 34)]
df_test_cols = df_test_cols1 + df_test_cols2
df_test.columns = df_test_cols
print(df_test.shape)
df_test.head()
X_test = df_test
print(X_test.shape)
column_trans = make_column_transformer((OneHotEncoder(handle_unknown='ignore'), ['item_category_id']), remainder='passthrough')
rfr = RandomForestRegressor(n_estimators=100)
pipeline = make_pipeline(column_trans, rfr)
(x_train, x_train_test, y_train, y_train_test) = train_test_split(X_train, Y_train, test_size=0.2, random_state=21)
print('Train set:', x_train.shape, y_train.shape)
print('Test set:', x_train_test.shape, y_train_test.shape)