import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import confusion_matrix
_input3 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
_input4 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
_input0 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
_input1 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
_input2 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
_input5 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
_input0['date'] = pd.to_datetime(_input0['date'], format='%d.%m.%Y')
pt = pd.pivot_table(_input0, index=['shop_id', 'item_id'], values='item_cnt_day', columns=['date_block_num'], aggfunc=np.sum, fill_value=0)
pt = pt.reset_index(inplace=False)
df = pd.merge(_input2, pt, on=['shop_id', 'item_id'], how='left')
df = df.fillna(0, inplace=False)
X_train = df.drop(columns=['shop_id', 'item_id', 'ID', 33], axis=1)
y_train = df[33]
X_test = df.drop(columns=['shop_id', 'item_id', 'ID', 0], axis=1)
X_test.columns = X_train.columns