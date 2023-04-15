import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
print(train_df)
train_df.isnull().sum()
train_df.info()
train_df['date'] = pd.to_datetime(train_df['date'], format='%d.%m.%Y')
print(train_df)
train_df['date'].max()
train_df['date'].min()
train_df['day'] = train_df['date'].dt.day
train_df['month'] = train_df['date'].dt.month
train_df['year'] = train_df['date'].dt.year
train_df['month_year'] = pd.to_datetime(train_df['date']).dt.to_period('M')
print(train_df.sort_values(by=['date']))
train_df[train_df['item_cnt_day'] <= 0]
train_df[train_df['item_cnt_day'] > 2000]
train_df['item_cnt_day'].max()
train_df['item_cnt_day'].min()
train_df[train_df['item_price'] <= 0]
train_df[train_df['item_price'] > 100000]
train_df['item_price'].max()
train_df['item_price'].min()
train_df = train_df[(train_df['item_price'] > 0) & (train_df['item_price'] < 2000)]
train_df = train_df[(train_df['item_cnt_day'] > 0) & (train_df['item_cnt_day'] < 100000)]
print(train_df)
test_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
print(test_df)
test_df['date_block_num'] = 34
test_data = test_df.drop(['ID'], axis=1)
print(test_data)
set(test_data['shop_id'].unique()).difference(set(train_df['shop_id'].unique()))
train_df['shop_id'].unique()
shops_data = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
shops_data = shops_data.sort_values(by=['shop_name'])
print(shops_data)
shops_data[['city', 'shop_name']] = shops_data['shop_name'].str.split(' ', 1, expand=True)
print(shops_data)
items_data = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
print(items_data)
items_data.item_category_id.value_counts()
All_df = train_df.merge(shops_data, left_on='shop_id', right_on='shop_id')
All_df = All_df.merge(items_data, left_on='item_id', right_on='item_id')
print(All_df)
item_cnt_shop_id = All_df.groupby(['date_block_num', 'shop_id']).agg({'item_cnt_day': 'sum', 'item_price': np.mean}).reset_index()
print(item_cnt_shop_id)
item_cnt_city = All_df.groupby(['date_block_num', 'city']).agg({'item_cnt_day': 'sum', 'item_price': np.mean}).reset_index()
print(item_cnt_city)
item_cnt_item = All_df.groupby(['date_block_num', 'item_id']).agg({'item_cnt_day': 'sum', 'item_price': np.mean}).reset_index()
print(item_cnt_item)
item_cnt_item_cat = All_df.groupby(['date_block_num', 'shop_id', 'item_category_id']).agg({'item_cnt_day': np.mean, 'item_price': np.mean}).reset_index()
print(item_cnt_item_cat)
sales_monthly = All_df.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_day': 'sum', 'item_price': np.mean}).reset_index()
print(sales_monthly)
sales_monthly_df = sales_monthly[['shop_id', 'item_id', 'date_block_num', 'item_cnt_day']]
print(sales_monthly_df)
test_df1 = test_data.merge(sales_monthly, on=['shop_id', 'item_id'], how='left', indicator=True)
print(test_df1)
df_test_only = test_df1[test_df1['_merge'] == 'left_only']
df_test_only = df_test_only.drop(['date_block_num_y', 'item_cnt_day', 'item_price', '_merge'], axis=1)
print(df_test_only)
df_test_only = df_test_only.merge(items_data, left_on='item_id', right_on='item_id')
df_test_only = df_test_only.drop(['item_name'], axis=1)
print(df_test_only)
test_merged_df = df_test_only.merge(item_cnt_item_cat, left_on=['shop_id', 'item_category_id'], right_on=['shop_id', 'item_category_id'])
test_merged_df = test_merged_df.drop(['date_block_num_x', 'item_category_id'], axis=1)
test_merged_df = test_merged_df.drop(['item_price'], axis=1)
print(test_merged_df)
df_total = pd.concat([sales_monthly_df, test_merged_df], ignore_index=True)
print(df_total)
df_total_34 = pd.concat([df_total, test_data], ignore_index=True)
df_total_34 = df_total_34.fillna(0)
print(df_total_34)
tmp2 = df_total_34.copy()
for i in range(1, 11):
    tmp2[str(i) + '_month_Ago_'] = tmp2.groupby(['shop_id', 'item_id'])['item_cnt_day'].shift(i)
tmp2 = tmp2.dropna().reset_index(drop=True)
print(tmp2)
tmp2[tmp2['date_block_num'] == 34]
for i in range(1, 35):
    train = tmp2[tmp2['date_block_num'] < i]
    val = tmp2[tmp2['date_block_num'] == i]
    (X_train, X_test) = (train.drop(['item_cnt_day', 'shop_id', 'item_id'], axis=1), val.drop(['item_cnt_day', 'shop_id', 'item_id'], axis=1))
    (y_train, y_test) = (train['item_cnt_day'].values, val['item_cnt_day'].values)
print(X_train)
print(y_train)
print(X_test)
print(y_test)
from numpy import absolute
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBRegressor
model3 = XGBRegressor()