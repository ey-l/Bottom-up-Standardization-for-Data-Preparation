import os
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style='darkgrid')
pd.set_option('display.float_format', lambda x: '%.2f' % x)
_input0 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
_input2 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
_input4 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
_input3 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
_input1 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
print('sales:', _input0.shape, 'test:', _input2.shape, 'items:', _input4.shape, 'item_cats:', _input3.shape, 'shop:', _input1.shape)
_input0['date'] = pd.to_datetime(_input0['date'], format='%d.%m.%Y')
_input0.head(3)
_input2.head(3)
_input2.shape
_input4.head(3)
_input3.head(3)
_input1.head(3)
_input0[_input0['item_price'] <= 0]
_input0[(_input0.shop_id == 32) & (_input0.item_id == 2973) & (_input0.date_block_num == 4)]
median = _input0[(_input0.shop_id == 32) & (_input0.item_id == 2973) & (_input0.date_block_num == 4) & (_input0.item_price > 0)].item_price.median()
_input0.loc[_input0.item_price < 0, 'item_price'] = median
dataset = _input0.pivot_table(index=['shop_id', 'item_id'], values=['item_cnt_day'], columns=['date_block_num'], fill_value=0, aggfunc='sum')
dataset
dataset = dataset.reset_index(inplace=False)
dataset.head()
dataset.shape
dataset = pd.merge(_input2, dataset, on=['item_id', 'shop_id'], how='left')
dataset
dataset = dataset.fillna(0, inplace=False)
dataset.head()
dataset = dataset.drop(['shop_id', 'item_id', 'ID'], inplace=False, axis=1)
dataset.head()
X_train = np.expand_dims(dataset.values[:, :-1], axis=2)
y_train = dataset.values[:, -1:]
X_test = np.expand_dims(dataset.values[:, 1:], axis=2)
print(X_train.shape, y_train.shape, X_test.shape)