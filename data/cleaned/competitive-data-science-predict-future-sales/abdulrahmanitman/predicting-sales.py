import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
item_cat = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
train.drop(labels=['date'], inplace=True, axis=1)
train.head(30)
train.describe()

import matplotlib.pyplot as plt
import seaborn as sns
fig = plt.figure(figsize=(16, 8))
sns.boxplot(x=train.item_cnt_day)
train = train[train['item_cnt_day'] < 500]
fig = plt.figure(figsize=(16, 8))
sns.boxplot(x=train.item_cnt_day)
train.info()
train.isnull().sum()
item_cnt_month = train.groupby(['date_block_num', 'shop_id'])[['item_cnt_day']].sum()
item_cnt_month.reset_index()
train = train.merge(item_cnt_month, on=['date_block_num', 'shop_id'])
train.sample(15)
train.rename(columns={'item_cnt_day_x': 'item_cnt_day', 'item_cnt_day_y': 'item_cnt_month'}, inplace=True)
train.sample(15)
X = train[['shop_id', 'item_id']]
Y = train['item_cnt_month']
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
(X_train, X_test, y_train, y_test) = train_test_split(X, Y, random_state=34)