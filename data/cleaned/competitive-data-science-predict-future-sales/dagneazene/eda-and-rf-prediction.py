import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
sales_train_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
items_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
item_cat_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
shops_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
sales_test_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
sales_train_df.head()
print(sales_train_df.nunique())
total_sales_month = sales_train_df.groupby('date_block_num', as_index=False)['item_cnt_day'].sum()
total_sales_month.head()
import matplotlib.pyplot as plt
total_sales_month.plot.bar(x='date_block_num', y='item_cnt_day', figsize=(12, 4))
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Month (date block no.)', fontsize=16)
plt.ylabel('total sales', fontsize=16)
plt.title('Total sales per month', fontsize=16)
sales_train_df['sales_income'] = sales_train_df['item_price'] * sales_train_df['item_cnt_day']
sales_train_df.head()
monthly_sales_price = sales_train_df.groupby(['date_block_num'], as_index=False)['sales_income'].sum()
monthly_sales_price.head()
import matplotlib.pyplot as plt
monthly_sales_price.plot.bar(x='date_block_num', y='sales_income', figsize=(12, 4))
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Month (date block no.)', fontsize=16)
plt.ylabel('total sales', fontsize=16)
plt.title('Total sales ($) per month', fontsize=16)
print(monthly_sales_price['sales_income'].corr(total_sales_month['item_cnt_day']))
sales_per_shop = sales_train_df.groupby(['shop_id', 'date_block_num'], as_index=False)['item_cnt_day'].sum()
sales_per_shop.head(10)
for i in range(2, 60, 5):
    import matplotlib.pyplot as plt
    df = sales_per_shop[sales_per_shop['shop_id'] == i]
    df.plot.bar(x='date_block_num', y='item_cnt_day', figsize=(8, 4))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Month (date block no.)', fontsize=12)
    plt.ylabel('total sales', fontsize=12)
    plt.title('Total sales per month; for shop %i' % i, fontsize=12)
items_df.head()
item_cat_df.head()
X_train = sales_train_df.merge(sales_per_shop, on=['date_block_num', 'shop_id'])
X_train.sample(10)
X_train.rename(columns={'item_cnt_day_x': 'item_cnt_day', 'item_cnt_day_y': 'item_cnt_month'}, inplace=True)
X_train.drop(labels=['date'], inplace=True, axis=1)
X_train.sample(5)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
X_training = X_train[['shop_id', 'item_id']]
Y = X_train['item_cnt_month']
(X_train, X_valid, y_train, y_valid) = train_test_split(X_training, Y, random_state=10)