import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
item_cat = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
sales_train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
sales_train.head(5)
sales_train['Month'] = [i.split('.')[1] for i in sales_train['date']]
sales_train['Year'] = [i.split('.')[2] for i in sales_train['date']]
sales_train['Month-Year'] = sales_train['Year'] + '-' + sales_train['Month']
dt = sales_train.groupby(['item_id', 'shop_id', 'Month-Year']).sum()[['item_cnt_day']].reset_index()
dt = dt[dt['Month-Year'] == '2015-10']
submission = pd.merge(test, dt, how='left', on=['item_id', 'shop_id']).fillna(0)
submission.columns = ['ID', 'shop_id', 'item_id', 'Month-Year', 'item_cnt_month']
submission['item_cnt_month'] = submission['item_cnt_month'].clip(0, 20)
submission.head()
