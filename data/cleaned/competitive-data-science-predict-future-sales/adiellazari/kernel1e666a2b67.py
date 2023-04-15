import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

data = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
submission = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
item_cats = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
print('Data Set')
print(data.shape)
print('Test Set')
print(test.shape)
data['date'] = pd.to_datetime(data['date'], format='%d.%m.%Y')
data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year
data = data.drop(['date', 'item_price'], axis=1)
data = data.groupby([c for c in data.columns if c not in ['item_cnt_day']], as_index=False)[['item_cnt_day']].sum()
data = data.rename(columns={'item_cnt_day': 'item_cnt_month'})
data.head()
shop_item_mean = data[['shop_id', 'item_id', 'item_cnt_month']].groupby(['shop_id', 'item_id'], as_index=False)[['item_cnt_month']].mean()
shop_item_mean = shop_item_mean.rename(columns={'item_cnt_month': 'item_cnt_month_mean'})
shop_prev_month = data[data['date_block_num'] == 33][['shop_id', 'item_id', 'item_cnt_month']]
shop_prev_month = shop_prev_month.rename(columns={'item_cnt_month': 'item_cnt_prev_mean'})
data = pd.merge(data, shop_prev_month, how='left', on=['shop_id', 'item_id']).fillna(0.0)
data = pd.merge(data, items, how='left', on='item_id')
data = pd.merge(data, item_cats, how='left', on='item_category_id')
data = pd.merge(data, shops, how='left', on='shop_id')
data.head()
test['month'] = 11
test['year'] = 2015
test['date_block_num'] = 34
test = pd.merge(test, shop_item_mean, how='left', on=['shop_id', 'item_id']).fillna(0.0)
test = pd.merge(test, shop_prev_month, how='left', on=['shop_id', 'item_id']).fillna(0.0)
test = pd.merge(test, items, how='left', on='item_id')
test = pd.merge(test, item_cats, how='left', on='item_category_id')
test = pd.merge(test, shops, how='left', on='shop_id')
test['item_cnt_month'] = 0.0
test.head()
from sklearn import preprocessing
for c in ['shop_name', 'item_name', 'item_category_name']:
    lbl = preprocessing.LabelEncoder()