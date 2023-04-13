import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
_input0 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
_input0.head(5)
print('Count of items this sold', len(_input0))
_input0.describe()
_input0.isnull().values.any()
print('Count of NaN: ', _input0.isnull().sum().sum())
pt = pd.pivot_table(_input0, index=['shop_id', 'item_id'], values='item_cnt_day', columns=['date_block_num'], aggfunc=np.sum, fill_value=0)
pt = pt.reset_index(inplace=False)
pt
pt.isnull().values.any()
print('Count of NaN: ', pt.isnull().sum().sum())
X = pt.drop(columns=['shop_id', 'item_id', 33], axis=1)
y = pt[33]
_input2 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
df_test = pd.merge(_input2, pt, on=['shop_id', 'item_id'], how='left')
df_test.head(5)
df_test = df_test.fillna(0, inplace=False)
df_test
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.1, random_state=123)
X_Test = df_test.drop(columns=['shop_id', 'item_id', 'ID', 0], axis=1)
X_Test.columns = X_train.columns