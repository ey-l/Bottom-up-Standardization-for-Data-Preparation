import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.metrics import mean_squared_error
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
test['date_block_num'] = 34
train_gp = train.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_day': ['sum']})
train_gp.columns = ['item_cnt_month']
train_gp.reset_index(inplace=True)
last_month = train_gp.copy()
last_month['date_block_num'] = last_month['date_block_num'] + 1
last_month = last_month.rename(columns={'item_cnt_month': 'last_month_cnt'})
train_gp = pd.merge(train_gp, last_month, on=['date_block_num', 'shop_id', 'item_id'], how='left')
train_gp = train_gp[train_gp.date_block_num != 0].fillna(0).reset_index(drop=True)
test = pd.merge(test, last_month, on=['date_block_num', 'shop_id', 'item_id'], how='left')
test = test.fillna(0).reset_index(drop=True)
y_true = np.where(train_gp.query('date_block_num == 33')['item_cnt_month'] > 20, 20, train_gp.query('date_block_num == 33')['item_cnt_month'])
y_pred = np.where(train_gp.query('date_block_num == 33')['last_month_cnt'] > 20, 20, train_gp.query('date_block_num == 33')['last_month_cnt'])
mean_squared_error(y_true, y_pred, squared=False)
from sklearn.ensemble import RandomForestRegressor
fea = ['last_month_cnt', 'shop_id', 'item_id']
train_gp.item_cnt_month = np.where(train_gp.item_cnt_month > 20, 20, train_gp.item_cnt_month)
X_train = train_gp.query('date_block_num < 33')[fea]
X_test = train_gp.query('date_block_num == 33')[fea]
y_train = train_gp.query('date_block_num < 33')['item_cnt_month']
y_test = train_gp.query('date_block_num == 33')['item_cnt_month']
rfr = RandomForestRegressor(max_depth=10, n_jobs=-1)