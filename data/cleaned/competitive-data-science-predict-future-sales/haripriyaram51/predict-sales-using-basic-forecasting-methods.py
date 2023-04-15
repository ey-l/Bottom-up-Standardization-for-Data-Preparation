import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
items_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
items_df.describe().T
item_categories_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
shops_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
sales_train_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
sales_train_df
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 5))
plt.hist(sales_train_df['item_id'])

sales_train_df.groupby(['item_id']).nunique()
sales_train_df[['shop_id', 'item_id', 'item_price', 'item_cnt_day']].corr()
sales_train_df.groupby(['shop_id']).nunique()
plt.figure(figsize=(6, 5))
plt.hist(sales_train_df['shop_id'])

sales_train_df.groupby(['shop_id', 'date_block_num']).sum('item_cnt_day')
sales_train_df_groupby = sales_train_df.groupby(['shop_id', 'item_id', 'date_block_num'])['item_cnt_day'].sum().reset_index()
sales_train_df_groupby
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
X = sales_train_df_groupby.drop(columns=['item_cnt_day', 'date_block_num'])
Y = sales_train_df_groupby['item_cnt_day']
(X_train, X_test, y_train, y_test) = train_test_split(X, Y, test_size=0.2, random_state=0)
X_train.head(2)
linear_regressor = LinearRegression()