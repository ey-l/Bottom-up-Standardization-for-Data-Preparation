import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from keras.layers import *
from keras.regularizers import l2
from keras.models import Model
from numpy import array
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
items_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
shops_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
cats_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv').set_index('ID')
train.head()
train.shape
shop_ids = test['shop_id'].unique()
item_ids = test['item_id'].unique()
filtered_train = train[train['shop_id'].isin(shop_ids)]
filtered_train = filtered_train[filtered_train['item_id'].isin(item_ids)]
print(filtered_train.shape)
filtered_train.head()
del filtered_train['date']
price_series = filtered_train[['item_price', 'item_id']].drop_duplicates(subset='item_id')
del filtered_train['item_price']
filtered_train.head()
train_df = filtered_train.groupby(['shop_id', 'item_id', 'date_block_num']).sum().reset_index()
train_df = train_df.rename(columns={'item_cnt_day': 'total_sales'}).sort_values(by=['date_block_num', 'item_id', 'shop_id'])
train_df.head()
train_df.shape
clf = DecisionTreeRegressor()