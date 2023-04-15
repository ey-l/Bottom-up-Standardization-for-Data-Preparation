import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

salesdata = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
salesdata.head()
salesdata.info()
itemdf = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
itemdf.head()
itemcat = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
iteminfo = pd.merge(itemdf, itemcat, on='item_category_id')
iteminfo.head()
salesitemdata = pd.merge(salesdata, iteminfo, on='item_id')
salesitemdata.head()
salesitemdata = salesitemdata.drop(['item_price', 'date', 'item_category_name', 'item_name'], axis=1)
salesitemdata.head()
salesitemdata['shop_id'] = salesitemdata['shop_id'].apply(int)
salesitemdata['item_id'] = salesitemdata['item_id'].apply(int)
salesitemdata['item_category_id'] = salesitemdata['item_category_id'].apply(int)
groupSalesDf = salesitemdata.groupby(['date_block_num', 'item_id', 'item_category_id', 'shop_id']).sum()
groupSalesDf['item_cnt_month'] = groupSalesDf['item_cnt_day']
groupSalesDf.drop('item_cnt_day', axis=1, inplace=True)
groupSalesDf.reset_index(inplace=True)
groupSalesDf.drop('item_category_id', axis=1, inplace=True)
groupSalesDf
groupSalesDf.info()
clippedSales = groupSalesDf.copy()
clippedSales['item_cnt_month'].clip(0, 20, inplace=True)
clippedSales['item_cnt_month'].max()
testdata = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv', index_col=0)
testdata
testdata['date_block_num'] = '34'
testdata
testdata['date_block_num'] = testdata['date_block_num'].apply(int)
testdata.info()
sns.lineplot(x='date_block_num', y='item_cnt_month', data=groupSalesDf[groupSalesDf['item_id'] == 22167])
X_train = groupSalesDf[0:1126386].drop('item_cnt_month', axis=1)
y_train = groupSalesDf['item_cnt_month'][0:1126386]
X_valid = groupSalesDf[1126386:].drop('item_cnt_month', axis=1)
y_valid = groupSalesDf['item_cnt_month'][1126386:]
from xgboost import XGBRegressor
model = XGBRegressor(early_stopping_rounds=5, eval_set=[(X_valid, y_valid)], objective='reg:squarederror', verbose=False)