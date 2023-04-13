import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
_input0 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
_input0.head()
_input0.info()
_input4 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
_input4.head()
_input3 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
iteminfo = pd.merge(_input4, _input3, on='item_category_id')
iteminfo.head()
salesitemdata = pd.merge(_input0, iteminfo, on='item_id')
salesitemdata.head()
salesitemdata = salesitemdata.drop(['item_price', 'date', 'item_category_name', 'item_name'], axis=1)
salesitemdata.head()
salesitemdata['shop_id'] = salesitemdata['shop_id'].apply(int)
salesitemdata['item_id'] = salesitemdata['item_id'].apply(int)
salesitemdata['item_category_id'] = salesitemdata['item_category_id'].apply(int)
groupSalesDf = salesitemdata.groupby(['date_block_num', 'item_id', 'item_category_id', 'shop_id']).sum()
groupSalesDf['item_cnt_month'] = groupSalesDf['item_cnt_day']
groupSalesDf = groupSalesDf.drop('item_cnt_day', axis=1, inplace=False)
groupSalesDf = groupSalesDf.reset_index(inplace=False)
groupSalesDf = groupSalesDf.drop('item_category_id', axis=1, inplace=False)
groupSalesDf
groupSalesDf.info()
clippedSales = groupSalesDf.copy()
clippedSales['item_cnt_month'] = clippedSales['item_cnt_month'].clip(0, 20, inplace=False)
clippedSales['item_cnt_month'].max()
_input2 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv', index_col=0)
_input2
_input2['date_block_num'] = '34'
_input2
_input2['date_block_num'] = _input2['date_block_num'].apply(int)
_input2.info()
sns.lineplot(x='date_block_num', y='item_cnt_month', data=groupSalesDf[groupSalesDf['item_id'] == 22167])
X_train = groupSalesDf[0:1126386].drop('item_cnt_month', axis=1)
y_train = groupSalesDf['item_cnt_month'][0:1126386]
X_valid = groupSalesDf[1126386:].drop('item_cnt_month', axis=1)
y_valid = groupSalesDf['item_cnt_month'][1126386:]