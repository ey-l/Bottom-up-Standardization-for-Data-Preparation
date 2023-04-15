import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        pass
items_data = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
shops_data = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
item_categories_data = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
train_data = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv', parse_dates=['date'], dayfirst=True)
test_data = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
train_data.head(3)
(train_data['date'].min(), train_data['date'].max())
item_categories_data.head(4)
items_data.head(5)
shops_data.head(5)
train_data1 = train_data.merge(items_data[['item_id', 'item_category_id']], how='left', on=['item_id'])
print('Total Number Unique Items', train_data1['item_id'].nunique())
print('Total Number Unique Items Category', train_data1['item_category_id'].nunique())
print('Total Number Unique Shops', train_data1['shop_id'].nunique())
print('Total Number Block Numbers', train_data1['date_block_num'].max())
test_data['date_block_num'] = train_data1['date_block_num'].max() + 1
test_data.shape
train_data2 = train_data1[['date_block_num', 'shop_id', 'item_id', 'item_cnt_day', 'item_price']]
train_data2.shape[0] + test_data.shape[0]
all_data = pd.concat([train_data2, test_data[['date_block_num', 'shop_id', 'item_id']]])
all_data.shape
train_data3 = all_data.groupby(['shop_id', 'item_id', 'date_block_num']).agg({'item_cnt_day': 'sum', 'item_price': 'mean'}).reset_index()
train_data3.head(3)
train_data3['last_day_sale'] = train_data3.groupby(['shop_id', 'item_id'])['item_cnt_day'].shift()
train_data3['day2_sale'] = train_data3.groupby(['shop_id', 'item_id'])['last_day_sale'].shift()
train_data3['day3_sale'] = train_data3.groupby(['shop_id', 'item_id'])['day2_sale'].shift()
train_data3['day4_sale'] = train_data3.groupby(['shop_id', 'item_id'])['day3_sale'].shift()
train_data3['day5_sale'] = train_data3.groupby(['shop_id', 'item_id'])['day4_sale'].shift()
train_data3['last_day_price'] = train_data3.groupby(['shop_id', 'item_id'])['item_price'].shift()
train_data3['day2_price'] = train_data3.groupby(['shop_id', 'item_id'])['last_day_price'].shift()
train_data3['day3_price'] = train_data3.groupby(['shop_id', 'item_id'])['day2_price'].shift()
train_data3['day4_price'] = train_data3.groupby(['shop_id', 'item_id'])['day3_price'].shift()
train_data3['day5_price'] = train_data3.groupby(['shop_id', 'item_id'])['day4_price'].shift()
train_data3['item_minprice'] = train_data3.groupby(['shop_id', 'item_id'])['item_price'].transform('min')
train_data3['item_maxprice'] = train_data3.groupby(['shop_id', 'item_id'])['item_price'].transform('max')
train_data3.corr()
import matplotlib.pyplot as plt
pjme_train = train_data3.loc[train_data3.date_block_num <= 32].copy()
pjme_test = train_data3.loc[train_data3.date_block_num == 33].copy()
pjme_future = train_data3.loc[train_data3.date_block_num == 34].copy()
pjme_train1 = pjme_train.dropna(subset=['last_day_sale', 'day2_sale', 'day3_sale', 'day4_sale', 'day5_sale'])
pjme_future.shape
features = ['shop_id', 'item_id', 'item_price', 'last_day_sale', 'day2_sale', 'day3_sale', 'day4_sale', 'day5_sale', 'item_minprice', 'item_maxprice']
target = 'item_cnt_day'
(X_train, y_train) = (pjme_train1[features], pjme_train1[target])
(X_test, y_test) = (pjme_test[features], pjme_test[target])
import xgboost as xgb
from xgboost import plot_importance, plot_tree
reg = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=50)