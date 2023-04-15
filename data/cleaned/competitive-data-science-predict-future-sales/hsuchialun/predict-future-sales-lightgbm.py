import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
item_categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
train_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
sample_sub = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
test_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
train_df.head()
train_df[train_df['item_cnt_day'] < 0] = 0
train_df = train_df[(train_df['item_price'] < 100000) & (train_df['item_price'] > 0)]
train_df = train_df[train_df['item_cnt_day'] < 1001]
train_df.drop(['item_price'], axis=1, inplace=True)
X = train_df.copy()
X.head()
X = X.groupby(['date_block_num', 'shop_id', 'item_id'], as_index=False).sum()
X = X.rename(columns={'item_cnt_day': 'item_cnt_month'})
X.head()
X = X.pivot_table(index=['shop_id', 'item_id'], columns='date_block_num', values='item_cnt_month', fill_value=0)
X.reset_index(inplace=True)
X.head()
test_df.head()
X_train = np.array(X.values[:, 0:-1])
Y_train = np.array(X.values[:, -1])
print(X_train.shape)
print(Y_train.shape)
from sklearn.linear_model import LinearRegression
model_linear = LinearRegression()