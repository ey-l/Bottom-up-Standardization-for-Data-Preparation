import pandas as pd
import numpy as np
from sklearn import tree
import os
category = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
sample = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
train = train.groupby(['date_block_num', 'shop_id', 'item_id'], as_index=False)['item_cnt_day'].sum().rename(columns={'item_cnt_day': 'item_cnt_month'})
train.head()
target = train['item_cnt_month'].values
features = train[['date_block_num', 'shop_id', 'item_id']].values
train.head()
my_tree_one = tree.DecisionTreeClassifier()