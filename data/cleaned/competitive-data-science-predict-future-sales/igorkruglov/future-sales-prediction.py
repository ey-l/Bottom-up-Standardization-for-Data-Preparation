import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
sample_submission = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
sample_submission.info()
sample_submission.head()
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
items.info()
items.head()
item_categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
item_categories.info()
item_categories.head()
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
shops.info()
shops.head()
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
train.info()
train.head()
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
test.info()
test.head()
df = train.groupby(['date_block_num', 'shop_id', 'item_id']).sum()
df = df.pivot_table(index=['shop_id', 'item_id'], columns='date_block_num', values='item_cnt_day', fill_value=0)
df.reset_index(inplace=True)
df
test = test.merge(df, on=['shop_id', 'item_id'], how='left').drop('ID', 1)
test.fillna(0, inplace=True)
X_train = df.drop(33, 1)
y_train = df[33]
X_test = test.drop(0, 1)
X_test_fin = X_test = test.drop(0, 1)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
(X_train, X_test, y_train, y_test) = train_test_split(X_train, y_train, test_size=0.2, random_state=4)
print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)
model = RandomForestRegressor(max_depth=8, n_estimators=200)