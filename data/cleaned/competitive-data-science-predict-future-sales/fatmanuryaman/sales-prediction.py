import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sales_train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
sales_train.head()
sales_train.info()
sales_train.isnull().sum()
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
items.head()
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
shops.head()
item_cat = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
item_cat.head()
sales_train_all1 = pd.merge(sales_train, shops, how='inner', on='shop_id')
sales_train_all1.head()
items_all = pd.merge(items, item_cat, how='inner', on='item_category_id')
items_all.head()
sales_train_all = pd.merge(sales_train_all1, items_all, how='inner', on='item_id')
sales_train_all.head()
sales_train_all.isnull().sum()
sales_train_all.info()
sales_train_all['date'] = pd.to_datetime(sales_train_all['date'], dayfirst=True)
sales_train_all['date'] = sales_train_all['date'].apply(lambda x: x.strftime('%Y-%m'))
sales_train_all.drop(columns=['item_category_name', 'item_name', 'shop_name', 'date_block_num'], inplace=True)
sales_train_all.head()
sales_train_all.head()
sales_train_all.tail()
sales_train_all.info()
corr = sales_train_all.corr()
(f, ax) = plt.subplots(figsize=(20, 20))
sns.heatmap(corr, annot=True)
data_sum = sales_train_all.groupby(['item_id', 'date', 'shop_id', 'item_price'], as_index=False)['item_cnt_day'].sum()
data_sum = data_sum.pivot_table(index=['shop_id', 'item_id'], columns='date', values='item_cnt_day', fill_value=0)
data_sum.reset_index(inplace=True)
data_sum.head()
test_data = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
test_data = pd.merge(test_data, data_sum, on=['shop_id', 'item_id'], how='left')
test_data.drop(['ID', '2013-01'], axis=1, inplace=True)
test_data = test_data.fillna(0)
test_data.head()
Y_train = data_sum['2014-08'].values
X_train = data_sum.drop(['2014-08'], axis=1)
X_test = test_data
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(X_train, Y_train, test_size=0.2, random_state=101)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
linear_reg = LinearRegression()