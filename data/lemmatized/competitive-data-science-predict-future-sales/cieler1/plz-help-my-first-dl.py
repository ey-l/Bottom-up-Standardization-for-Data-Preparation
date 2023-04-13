import pandas as pd
import numpy as np
from sklearn import tree
import os
_input3 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
_input4 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
_input0 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
_input5 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
_input1 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
_input2 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
_input0 = _input0.groupby(['date_block_num', 'shop_id', 'item_id'], as_index=False)['item_cnt_day'].sum().rename(columns={'item_cnt_day': 'item_cnt_month'})
_input0.head()
target = _input0['item_cnt_month'].values
features = _input0[['date_block_num', 'shop_id', 'item_id']].values
_input0.head()
my_tree_one = tree.DecisionTreeClassifier()