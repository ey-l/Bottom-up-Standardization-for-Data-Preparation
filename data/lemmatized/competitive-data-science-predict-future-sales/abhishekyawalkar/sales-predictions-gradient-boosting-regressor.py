import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
_input4 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
_input4.head()
_input3 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
_input3.head()
_input0 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
_input0.head()
_input0 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
_input0.head()
_input0.shape
_input2 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
Id = _input2['ID']
_input2.head()
_input2.shape
_input0.info()
_input0['date'] = pd.to_datetime(_input0['date'])
_input0.isna().sum()
_input0['date'] = _input0['date'].apply(lambda x: x.strftime('%Y-%m'))
_input0.head().sort_values(by='date')
_input0 = _input0.drop(['date_block_num', 'item_price'], axis=1, inplace=False)
_input0.head().sort_values(by='date')
df = _input0.groupby(['date', 'shop_id', 'item_id']).sum()
df
df = _input0.pivot_table(index=['shop_id', 'item_id'], columns='date', values='item_cnt_day', fill_value=0)
df = df.reset_index(inplace=False)
df.head()
test_df = pd.merge(_input2, df, on=['shop_id', 'item_id'], how='left')
test_df = test_df.drop(['ID', '2013-01'], axis=1, inplace=False)
test_df = test_df.fillna(0)
test_df.head()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
Y = df['2015-10'].values
X = df.drop(['2015-10'], axis=1)
test_full = test_df
(X_train, X_test, y_train, y_test) = train_test_split(X, Y, test_size=0.2, random_state=42)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
lr = LinearRegression()