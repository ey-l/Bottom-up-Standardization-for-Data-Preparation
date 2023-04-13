import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
_input0 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
_input0.head()
_input0.info()
_input0.isnull().sum()
_input4 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
_input4.head()
_input1 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
_input1.head()
_input3 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
_input3.head()
sales_train_all1 = pd.merge(_input0, _input1, how='inner', on='shop_id')
sales_train_all1.head()
items_all = pd.merge(_input4, _input3, how='inner', on='item_category_id')
items_all.head()
sales_train_all = pd.merge(sales_train_all1, items_all, how='inner', on='item_id')
sales_train_all.head()
sales_train_all.isnull().sum()
sales_train_all.info()
sales_train_all['date'] = pd.to_datetime(sales_train_all['date'], dayfirst=True)
sales_train_all['date'] = sales_train_all['date'].apply(lambda x: x.strftime('%Y-%m'))
sales_train_all = sales_train_all.drop(columns=['item_category_name', 'item_name', 'shop_name', 'date_block_num'], inplace=False)
sales_train_all.head()
sales_train_all.head()
sales_train_all.tail()
sales_train_all.info()
corr = sales_train_all.corr()
(f, ax) = plt.subplots(figsize=(20, 20))
sns.heatmap(corr, annot=True)
data_sum = sales_train_all.groupby(['item_id', 'date', 'shop_id', 'item_price'], as_index=False)['item_cnt_day'].sum()
data_sum = data_sum.pivot_table(index=['shop_id', 'item_id'], columns='date', values='item_cnt_day', fill_value=0)
data_sum = data_sum.reset_index(inplace=False)
data_sum.head()
_input2 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
_input2 = pd.merge(_input2, data_sum, on=['shop_id', 'item_id'], how='left')
_input2 = _input2.drop(['ID', '2013-01'], axis=1, inplace=False)
_input2 = _input2.fillna(0)
_input2.head()
Y_train = data_sum['2014-08'].values
X_train = data_sum.drop(['2014-08'], axis=1)
X_test = _input2
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(X_train, Y_train, test_size=0.2, random_state=101)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
linear_reg = LinearRegression()