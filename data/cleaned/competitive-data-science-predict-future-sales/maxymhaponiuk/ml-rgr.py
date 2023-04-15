import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
train.head()
train.info()
train.describe()
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
test.head()
test.info()
test.describe()
item_categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
item_categories.head()
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
shops.head()
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
items.head()
sample_submission = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
sample_submission.head()

def missing(df):
    missing_df = pd.DataFrame({'missing_count': df.isnull().sum(), 'missing_part': df.isnull().sum() / len(df)})
    missing_df = missing_df[missing_df['missing_count'] != 0]
    return missing_df
missing(train)
missing(test)
features_to_check = ['item_cnt_day', 'item_price', 'date_block_num']
for feature in features_to_check:
    plt.figure(figsize=(20, 3))
    sns.boxplot(data=train[feature], orient='h')
    plt.title(feature)

train = train[train['item_cnt_day'] > 0]
train = train[train['item_price'] > 0]
train.duplicated().sum()
train = train.drop_duplicates()
df_to_plot = train.groupby(['date_block_num']).agg({'item_cnt_day': 'sum'})
plt.plot(df_to_plot['item_cnt_day'])
df_to_plot.sort_values(by=['item_cnt_day'], ascending=False).head()
train['item_cnt_day_sum'] = train.item_price * train.item_cnt_day
df_to_plot = train.groupby(['date_block_num']).agg({'item_cnt_day_sum': 'sum'})
plt.plot(df_to_plot['item_cnt_day_sum'])
top_categories = train.sort_values(by=['item_cnt_day'], ascending=False).head(10)
top_categories.plot.bar(x='item_id', y='item_cnt_day')
top_categories_items = top_categories['item_id'].values
item_categories[item_categories['item_category_id'].isin(items[items['item_id'].isin(top_categories_items)]['item_category_id'].values)]
shop_by_cnt = train.groupby('shop_id').agg({'item_cnt_day': 'sum'}).sort_values(by=['item_cnt_day'], ascending=False)
shop_by_cnt.plot.bar(figsize=(20, 10))
shop_by_cnt_sum = train.groupby('shop_id').agg({'item_cnt_day_sum': 'sum'}).sort_values(by=['item_cnt_day_sum'], ascending=False)
shop_by_cnt_sum.plot.bar(figsize=(20, 10))
shops[shops['shop_id'].isin(shop_by_cnt_sum.head(5).index)]
shops[shops['shop_id'].isin(shop_by_cnt.head(5).index)]
train
train.drop(columns=['item_cnt_day_sum'], inplace=True)
train['month_num'] = train['date'].str.split('.').str.get(1)
train.drop(columns=['date'], inplace=True)
train['month_num'] = train['month_num'].apply(pd.to_numeric)
train['month_num'] = train['month_num'].astype('int16')
train.head(5)
train_grpd = train.drop(columns=['month_num']).groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_price': 'mean', 'item_cnt_day': 'sum'})
train_grpd.head(5)
train_grpd = train_grpd.join(shops.set_index('shop_id'), how='inner')
train_grpd
train_grpd = train_grpd.join(items.set_index('item_id'), how='inner')
train_grpd
train_grpd = train_grpd.join(item_categories.set_index('item_category_id'), on='item_category_id', how='inner')
train_grpd
train_grpd['location'] = train_grpd['shop_name'].str.split().str.get(0)
train_grpd.head(5)
train_grpd.drop(columns=['item_category_name', 'item_name', 'shop_name'], inplace=True)
train_grpd.head(5)
months_numbers = train[['date_block_num', 'month_num']].drop_duplicates()
train_grpd = train_grpd.join(months_numbers.set_index('date_block_num'), how='inner')
train_grpd
train_grpd.rename(columns={'item_cnt_day': 'item_cnt_month', 'item_price': 'item_price_mean'}, inplace=True)
train_grpd
train_grpd.reset_index()['item_cnt_month'].plot(figsize=(10, 5))
test.head()
len(test)
test['date_block_num'] = 34
test['month_num'] = 11
test.head()
test_df = test.join(shops.set_index('shop_id'), on='shop_id', how='inner')
test_df.head()
test_df = test_df.join(items.set_index('item_id'), on='item_id', how='inner')
test_df.head()
test_df = test_df.join(item_categories.set_index('item_category_id'), on='item_category_id', how='inner')
test_df.head()
test_df['location'] = test_df['shop_name'].str.split().str.get(0)
test_df.head(5)
test_df.drop(columns=['item_category_name', 'item_name', 'shop_name'], inplace=True)
test_df.head(5)
train_grpd.sample(10)
test_df
test_df.join(train_grpd.reset_index(level=0)['item_price_mean'], on=['shop_id', 'item_id'], how='inner')
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import RidgeCV
train_Y = train_grpd['item_cnt_month'].values
train_X = train_grpd.reset_index().drop(columns=['item_cnt_month', 'item_price_mean'])
train_X['location'] = train_X['location'].astype('category')
test_X = test_df.drop(columns=['ID'])
print(train_X.columns)
print(test_X.columns)
train_X = pd.get_dummies(train_X)
test_X = pd.get_dummies(test_df)
print(train_X.shape)
print(test_X.shape)
(train_X, test_X) = train_X.align(test_X, join='inner', axis=1)
print(train_X.shape)
print(test_X.shape)
(x_train, x_test, y_train, y_test) = train_test_split(train_X, train_Y)
regr = LinearRegression()