import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
data = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
data.head(5)
print('Count of items this sold', len(data))
data.describe()
data.isnull().values.any()
print('Count of NaN: ', data.isnull().sum().sum())
pt = pd.pivot_table(data, index=['shop_id', 'item_id'], values='item_cnt_day', columns=['date_block_num'], aggfunc=np.sum, fill_value=0)
pt.reset_index(inplace=True)
pt
pt.isnull().values.any()
print('Count of NaN: ', pt.isnull().sum().sum())
X = pt.drop(columns=['shop_id', 'item_id', 33], axis=1)
y = pt[33]
data_test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
df_test = pd.merge(data_test, pt, on=['shop_id', 'item_id'], how='left')
df_test.head(5)
df_test.fillna(0, inplace=True)
df_test
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.1, random_state=123)
X_Test = df_test.drop(columns=['shop_id', 'item_id', 'ID', 0], axis=1)
X_Test.columns = X_train.columns
from xgboost import XGBRegressor
from numpy import absolute
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
model = XGBRegressor(max_depth=30, n_estimators=50)