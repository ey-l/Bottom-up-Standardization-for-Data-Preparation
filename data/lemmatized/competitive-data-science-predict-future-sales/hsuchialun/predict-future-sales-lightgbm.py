import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
_input3 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
_input4 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
_input0 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
_input5 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
_input1 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
_input2 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
_input0.head()
_input0[_input0['item_cnt_day'] < 0] = 0
_input0 = _input0[(_input0['item_price'] < 100000) & (_input0['item_price'] > 0)]
_input0 = _input0[_input0['item_cnt_day'] < 1001]
_input0 = _input0.drop(['item_price'], axis=1, inplace=False)
X = _input0.copy()
X.head()
X = X.groupby(['date_block_num', 'shop_id', 'item_id'], as_index=False).sum()
X = X.rename(columns={'item_cnt_day': 'item_cnt_month'})
X.head()
X = X.pivot_table(index=['shop_id', 'item_id'], columns='date_block_num', values='item_cnt_month', fill_value=0)
X = X.reset_index(inplace=False)
X.head()
_input2.head()
X_train = np.array(X.values[:, 0:-1])
Y_train = np.array(X.values[:, -1])
print(X_train.shape)
print(Y_train.shape)
from sklearn.linear_model import LinearRegression
model_linear = LinearRegression()