

import lightgbm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor, BaggingRegressor
from catboost import CatBoostRegressor
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
items_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
samples_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
itemsCat_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
sales_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
shopes_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
test_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
print('---------items----------')
items_df.info()
print('--------samples---------')
samples_df.info()
print('--------items Category---------')
itemsCat_df.info()
print('--------Sales---------')
sales_df.info()
print('--------Shopes---------')
shopes_df.info()
print('--------Test---------')
test_df.info()
sales_df['month'] = pd.DatetimeIndex(pd.to_datetime(sales_df['date'], format='%d.%m.%Y')).month
sales_df['year'] = pd.DatetimeIndex(pd.to_datetime(sales_df['date'], format='%d.%m.%Y')).year
sales_df['day'] = pd.DatetimeIndex(pd.to_datetime(sales_df['date'], format='%d.%m.%Y')).day
sales_df.head(10)
print('unique item', len(sales_df.item_id.unique()))
print('unique shop', len(sales_df.shop_id.unique()))
df = sales_df[sales_df['year'] == 2013][['month', 'item_cnt_day']].groupby(['month']).sum().reset_index()
plt.plot(df['month'], df['item_cnt_day'])
df = sales_df[sales_df['year'] == 2014][['month', 'item_cnt_day']].groupby(['month']).sum().reset_index()
plt.plot(df['month'], df['item_cnt_day'])
df = sales_df[sales_df['year'] == 2015][['month', 'item_cnt_day']].groupby(['month']).sum().reset_index()
plt.plot(df['month'], df['item_cnt_day'])
train_df = sales_df[['item_id', 'shop_id', 'month', 'year', 'date_block_num', 'item_cnt_day']].groupby(['item_id', 'shop_id', 'month', 'year', 'date_block_num']).sum().reset_index()
train_df.rename(columns={'item_cnt_day': 'item_cnt_month'}, inplace=True)
train_df.head(5)
train_df.boxplot(column=['item_cnt_month'])
train_df['item_cnt_month'] = train_df['item_cnt_month'].clip(0, 1200)
train_df.boxplot(column=['item_cnt_month'])
test_df['year'] = 2015
test_df['month'] = 11
test_df['date_block_num'] = 34
test_df.head(5)
features = ['item_id', 'shop_id', 'month', 'year', 'date_block_num']
(train_X, val_X, train_y, val_y) = train_test_split(train_df[features], train_df['item_cnt_month'], test_size=0.1, random_state=0)