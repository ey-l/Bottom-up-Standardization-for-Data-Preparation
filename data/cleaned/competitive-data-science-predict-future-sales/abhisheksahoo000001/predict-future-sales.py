import pandas as pd
import numpy as np
sales = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
sales.head()
sales.describe()
sales.isnull().sum()
sales.drop(['date_block_num', 'item_price'], axis=1, inplace=True)
sales['date'] = pd.to_datetime(sales['date'], dayfirst=True)
sales['date'] = sales['date'].apply(lambda x: x.strftime('%Y-%m'))
sales['date']
sales.head()
sales = sales.groupby(['date', 'shop_id', 'item_id']).sum()
sales = sales.pivot_table(index=['shop_id', 'item_id'], columns='date', values='item_cnt_day', fill_value=0)
sales.reset_index(inplace=True)
sales.head()
y_train = sales['2015-10'].values
x_train = sales.drop(['2015-10'], axis=1)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)