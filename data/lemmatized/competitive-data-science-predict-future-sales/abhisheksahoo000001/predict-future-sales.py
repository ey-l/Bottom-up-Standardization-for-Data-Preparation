import pandas as pd
import numpy as np
_input0 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
_input0.head()
_input0.describe()
_input0.isnull().sum()
_input0 = _input0.drop(['date_block_num', 'item_price'], axis=1, inplace=False)
_input0['date'] = pd.to_datetime(_input0['date'], dayfirst=True)
_input0['date'] = _input0['date'].apply(lambda x: x.strftime('%Y-%m'))
_input0['date']
_input0.head()
_input0 = _input0.groupby(['date', 'shop_id', 'item_id']).sum()
_input0 = _input0.pivot_table(index=['shop_id', 'item_id'], columns='date', values='item_cnt_day', fill_value=0)
_input0 = _input0.reset_index(inplace=False)
_input0.head()
y_train = _input0['2015-10'].values
x_train = _input0.drop(['2015-10'], axis=1)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)