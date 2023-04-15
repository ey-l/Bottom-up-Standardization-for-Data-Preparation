import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
path = '_data/input/competitive-data-science-predict-future-sales/'
sales_train = pd.read_csv(path + 'sales_train.csv')
sales_train.head(5)
sales_train.describe()
item_categories = pd.read_csv(path + 'item_categories.csv')
item_categories.head()
shops = pd.read_csv(path + 'shops.csv')
shops.head()
items = pd.read_csv(path + 'items.csv')
items.head()
test = pd.read_csv(path + 'test.csv')
test.head()
sales_train['total_price'] = sales_train.item_cnt_day * sales_train.item_price
sales_train_shop_item = sales_train.groupby(by=['date_block_num', 'shop_id', 'item_id'])[['item_cnt_day', 'total_price']].sum()
sales_train_shop_item = sales_train_shop_item.reset_index()
sales_train_shop_item.rename(columns={'item_cnt_day': 'item_cnt_month'}, inplace=True)
sales_train.head()
sales_train_shop_item.describe()
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
X = sales_train_shop_item[['date_block_num', 'shop_id', 'item_id']]
y = sales_train_shop_item.total_price