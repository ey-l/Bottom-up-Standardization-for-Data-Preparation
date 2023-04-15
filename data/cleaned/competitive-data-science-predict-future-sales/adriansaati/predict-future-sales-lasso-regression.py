import pandas as pd
item_categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
sales_train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
type(sales_train['date'][0])
from datetime import datetime

def timestamp(x):
    return datetime.strptime(x, '%d.%m.%Y')
sales_train['date'] = sales_train['date'].apply(timestamp)
type(sales_train['date'][0])
sales_train
items.drop('item_name', 1, inplace=True)
items
sales_train.sort_values(by='date', inplace=True)
sales_train.reset_index(drop=True, inplace=True)
sales_train
sales_train = sales_train[['shop_id', 'item_id', 'date', 'date_block_num', 'item_price', 'item_cnt_day']]
sales_train.reset_index(drop=True, inplace=True)
sales_train
dataset = sales_train.pivot_table(index=['shop_id', 'item_id'], values=['item_cnt_day'], columns=['date_block_num'], fill_value=0)
dataset.reset_index(inplace=True)
dataset
test
dataset = pd.merge(test, dataset, on=['shop_id', 'item_id'], how='left')
dataset
dataset.fillna(0, inplace=True)
dataset
dataset.drop(labels=['shop_id', 'ID', 'item_id'], inplace=True, axis=1)
dataset
X_train = dataset.iloc[:, :-1]
y_train = dataset.iloc[:, -1:]
X_test = dataset.iloc[:, 1:]
print(X_train.shape, y_train.shape, X_test.shape)
X_train
y_train
X_test
from sklearn.linear_model import LinearRegression
regression = LinearRegression()