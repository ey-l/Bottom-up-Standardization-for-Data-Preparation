import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
sales = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv', parse_dates=['date'], infer_datetime_format=True, dayfirst=True)
sales.head()
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
test.head()
sales_column = ['date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day']
sales.drop_duplicates(sales_column, keep='first', inplace=True)
sales.reset_index(drop=True, inplace=True)
test_column = ['shop_id', 'item_id']
test.drop_duplicates(test_column, keep='first', inplace=True)
test.reset_index(drop=True, inplace=True)
sales.loc[sales.item_price < 0, 'item_price'] = 0
sales['item_cnt_day'] = sales['item_cnt_day'].clip(0, 1000)
sales['item_price'].max()
sales['item_price'] = sales['item_price'].clip(0, 300000)
df = sales.groupby([sales.date.apply(lambda x: x.strftime('%Y-%m')), 'item_id', 'shop_id']).sum().reset_index()
df.columns
df = df[['date', 'item_id', 'shop_id', 'item_cnt_day']]
df = df.pivot_table(index=['item_id', 'shop_id'], columns='date', values='item_cnt_day', fill_value=0).reset_index()
df
data = pd.merge(test, df, on=['item_id', 'shop_id'], how='left').fillna(0)
y = y = data.iloc[:, -1:]
x = data.iloc[:, 3:]
x.drop(['2015-10'], axis=1, inplace=True)
y
x
x = x.values
y = y.values.reshape(-1, 1)
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor()