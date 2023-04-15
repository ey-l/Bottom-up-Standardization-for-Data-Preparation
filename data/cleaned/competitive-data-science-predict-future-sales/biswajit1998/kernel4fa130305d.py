import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
path = '_data/input/competitive-data-science-predict-future-sales/'
sales_train = pd.read_csv(path + 'sales_train.csv')
sales_train.head(5)
item_categories = pd.read_csv(path + 'item_categories.csv')
item_categories.head()
shops = pd.read_csv(path + 'shops.csv')
shops.head()
items = pd.read_csv(path + 'items.csv')
items.head()
test = pd.read_csv(path + 'test.csv')
test.head()
test['date_block_num'] = 34
test = test.merge(items, how='left', on='item_id')
test.drop(['item_name'], axis=1, inplace=True)
test.head()
sales_train = sales_train.merge(shops, how='left', on='shop_id')
sales_train = sales_train.merge(items, how='left', on='item_id')
sales_train = sales_train.merge(item_categories, how='left', on='item_category_id')
sales_train.head()
sales_train['total_price_per_day'] = sales_train.item_cnt_day * sales_train.item_price
sales_train.head()
sales_train[['day', 'month', 'year']] = sales_train.date.str.split('.', expand=True)
sales_train.head()
sales_train_shop_item = sales_train.groupby(by=['date_block_num', 'shop_id', 'item_id', 'item_category_id'])[['item_cnt_day', 'total_price_per_day']].sum()
sales_train_shop_item = sales_train_shop_item.reset_index()
sales_train_shop_item.rename(columns={'item_cnt_day': 'item_cnt_month'}, inplace=True)
sales_train_shop_item.rename(columns={'total_price_per_day': 'total_price_per_month'}, inplace=True)
sales_train_shop_item.head()
sales_min = sales_train.groupby(by=['date_block_num', 'shop_id', 'item_id', 'item_category_id'])[['item_cnt_day', 'total_price_per_day']].min()
sales_min = sales_min.reset_index()
sales_min.head()
sales_train_shop_item['min_item_cnt_month'] = sales_min['item_cnt_day']
sales_train_shop_item['min_price_per_month'] = sales_min['total_price_per_day']
sales_train_shop_item.head()
sales_max = sales_train.groupby(by=['date_block_num', 'shop_id', 'item_id', 'item_category_id'])[['item_cnt_day', 'total_price_per_day']].max()
sales_max = sales_max.reset_index()
sales_train_shop_item['max_item_cnt_month'] = sales_max['item_cnt_day']
sales_train_shop_item['max_price_per_month'] = sales_max['total_price_per_day']
sales_train_shop_item.head()
sales_avg = sales_train.groupby(by=['date_block_num', 'shop_id', 'item_id', 'item_category_id'])[['item_cnt_day', 'total_price_per_day']].mean()
sales_avg = sales_avg.reset_index()
sales_train_shop_item['avg_item_cnt_month'] = sales_avg['item_cnt_day']
sales_train_shop_item['avg_price_per_month'] = sales_avg['total_price_per_day']
test.head()
sales_train_shop_item.head()
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
X = sales_train_shop_item[['date_block_num', 'shop_id', 'item_id']]
y = sales_train_shop_item['total_price_per_month']
poly = PolynomialFeatures(degree=3)
X_ = poly.fit_transform(X)