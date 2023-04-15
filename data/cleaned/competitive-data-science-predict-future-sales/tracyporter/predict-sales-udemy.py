import pandas as pd
import numpy as np
from pandas import read_csv
from pandas import datetime
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
submission = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
item_cat = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
submission
train['date'] = pd.to_datetime(train['date'], format='%d.%m.%Y')
train
dataset = train.pivot_table(index=['shop_id', 'item_id'], values=['item_cnt_day'], columns=['date_block_num'], fill_value=0, aggfunc='sum')
dataset.reset_index(inplace=True)
dataset
test = test.drop(['ID'], axis=1)
test
dataset = pd.merge(test, dataset, on=['item_id', 'shop_id'], how='left')
dataset
dataset.isnull().sum().sum()
dataset.fillna(0, inplace=True)
dataset.isnull().sum().sum()
dataset.drop(['shop_id', 'item_id'], inplace=True, axis=1)
dataset
y_train = dataset.iloc[:, -1:]
X_train = dataset.iloc[:, :-1]
X_test = dataset.iloc[:, 1:]
print(X_train.shape, y_train.shape, X_test.shape)
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor