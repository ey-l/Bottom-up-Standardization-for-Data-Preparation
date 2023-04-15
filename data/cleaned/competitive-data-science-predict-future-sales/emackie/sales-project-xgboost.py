import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
pd.set_option('max_colwidth', 400)
train_data = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
items_info = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
if False:
    train_data.info()
    items_info.info()
    category_info.info()
    shops_info.info()
train_data.head()
train_data['date'] = pd.to_datetime(train_data['date'], dayfirst=True)
items_info = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
grouped_data = train_data.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_price': ['mean', 'max', 'min'], 'item_cnt_day': ['sum', 'max', 'min', 'median']})
grouped_data = grouped_data.reset_index()
grouped_data = grouped_data.merge(items_info[['item_id', 'item_category_id']].drop_duplicates(), left_on='item_id', right_on='item_id', how='left')
grouped_data.drop('item_id', axis=1, inplace=True)
grouped_data.columns = ['date_block_num', 'shop_id', 'item_id', 'item_price_mean', 'item_price_max', 'item_price_min', 'item_cnt_day_sum', 'item_cnt_day_max', 'item_cnt_day_min', 'item_cnt_day_median', 'item_category_id']
t_minus1_data = grouped_data.loc[grouped_data.date_block_num != 33, ['date_block_num', 'shop_id', 'item_id', 'item_price_mean', 'item_cnt_day_sum', 'item_cnt_day_median']]
t_minus1_data['date_block_num'] += 1
driver_data = grouped_data[grouped_data.date_block_num != 0]
driver_data = driver_data.merge(t_minus1_data, left_on=['date_block_num', 'shop_id', 'item_id'], right_on=['date_block_num', 'shop_id', 'item_id'], suffixes=('', '_tminus1'))
for col in ['item_price_mean', 'item_cnt_day_sum', 'item_cnt_day_median']:
    driver_data[col + '_tminus1'] = np.log(driver_data[col + '_tminus1']) - np.log(driver_data[col])
submission_data = driver_data.copy()
response_data = grouped_data.loc[grouped_data.date_block_num != 0, ['date_block_num', 'shop_id', 'item_id', 'item_cnt_day_sum']]
driver_data = driver_data[driver_data.date_block_num != 33]
response_data['date_block_num'] -= 1
driver_data = driver_data.merge(response_data, left_on=['date_block_num', 'shop_id', 'item_id'], right_on=['date_block_num', 'shop_id', 'item_id'], suffixes=('', '_realised'))
xgb_model = XGBRegressor(colsample_bylevel=0.8, colsample_bynode=0.8, colsample_bytree=0.8, subsample=0.8, gamma=10, learning_rate=0.1, max_depth=12, min_child_weight=1, n_estimators=500, random_state=27, reg_alpha=50, reg_lambda=200)
training_cols = driver_data.drop(['date_block_num', 'shop_id', 'item_id', 'item_cnt_day_sum_realised'], axis=1).columns
(X, y) = (driver_data.drop(['date_block_num', 'shop_id', 'item_id', 'item_cnt_day_sum_realised'], axis=1).values, driver_data['item_cnt_day_sum_realised'].values)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=27, shuffle=True)
eval_set = [(X_test, y_test)]