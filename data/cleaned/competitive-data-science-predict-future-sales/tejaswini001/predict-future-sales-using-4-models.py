import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
train.head()
test.head()
train['item_cnt_day'] = abs(train['item_cnt_day'])
train.drop(columns=['date', 'item_price'], axis=1, inplace=True)
train.drop_duplicates(inplace=True, keep='first', ignore_index=True)
train
train1 = pd.pivot_table(train, index=['shop_id', 'item_id'], columns='date_block_num', values='item_cnt_day', aggfunc=np.sum).reset_index()
train1
test1 = test.merge(train1, how='left', on=['shop_id', 'item_id']).drop(columns=['shop_id', 'item_id']).fillna(value=0)
test1
x_train = test1.iloc[:, 1:-1]
y_train = test1.iloc[:, -1]
x_test = test1.iloc[:, 2:]
x_train
y_train
x_test
lr = LinearRegression()