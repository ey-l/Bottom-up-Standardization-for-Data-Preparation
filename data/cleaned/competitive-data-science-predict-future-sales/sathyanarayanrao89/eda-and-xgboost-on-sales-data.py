import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn import preprocessing
from sklearn.decomposition import PCA
train_data = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test_data = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
submission = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
item_categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
train_data.info()
train_data[['shop_id', 'item_id', 'item_price', 'item_cnt_day']].hist(bins=150, figsize=(9, 7), grid=False)
(fig1, ax) = plt.subplots(1, 2, figsize=(10, 5), dpi=100)
train_data.item_price.plot(ax=ax[0])
train_data.item_cnt_day.plot(ax=ax[1])
train_data.describe()
train_data.drop(train_data[train_data.item_price > 5 * train_data.item_price.std()].index, inplace=True)
train_data.drop(train_data[train_data.item_cnt_day > 5 * train_data.item_cnt_day.std()].index, inplace=True)
train_data.drop(train_data[train_data.item_price < 0].index, inplace=True)
train_data.drop(train_data[train_data.item_cnt_day < 0].index, inplace=True)
ID = test_data['ID']
test_data = test_data.drop(['ID'], axis=1)
train_data.info()
train_data[['shop_id', 'item_id', 'item_price', 'item_cnt_day']].hist(bins=150, figsize=(9, 7), grid=False)
train_data.info()
shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'
shops['city'] = shops.shop_name.str.split(' ').map(lambda x: x[0])
from sklearn.preprocessing import LabelEncoder
shops['city'] = LabelEncoder().fit_transform(shops.city)
dataFrame = train_data.copy()
dataFrame['date'] = pd.to_datetime(dataFrame['date'], format='%d.%m.%Y')
dataFrame['month'] = dataFrame['date'].dt.month
dataFrame = dataFrame[['month', 'date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day']].groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_price': 'mean', 'item_cnt_day': 'sum', 'month': 'min'}).reset_index()
dataFrame.rename(columns={'item_cnt_day': 'item_cnt_month'}, inplace=True)
dataFrame = pd.merge(dataFrame, items[['item_id', 'item_category_id']], how='inner')
dataFrame = pd.merge(dataFrame, shops[['shop_id', 'city']], how='inner')
dataFrame = dataFrame.sort_values(by=['date_block_num'], ascending=True).reset_index(drop=True)
dataFrame = dataFrame.pivot_table(index=['shop_id', 'item_id', 'city', 'item_category_id'], values=['item_cnt_month'], columns=['date_block_num'])
dataFrame.reset_index(inplace=True)
dataFrame.columns
test_data.columns
test_data = pd.merge(test_data, items[['item_id', 'item_category_id']], how='inner')
test_data = pd.merge(test_data, shops[['shop_id', 'city']], how='inner')
test_data.columns
dataFrame = pd.merge(test_data, dataFrame, on=['shop_id', 'item_id', 'city', 'item_category_id'], how='left')
dataFrame.fillna(0, inplace=True)
columns = list(dataFrame.columns[0:4])
for i in range(34):
    columns.append('time' + str(i))
dataFrame = dataFrame.rename(columns=dict(zip(list(dataFrame.columns), columns), inplace=True))
dataFrame.columns
test_data.info()
test_data.head()
dataFrame.head()
dataFrame = dataFrame.drop(list(dataFrame.columns[0:4]), axis=1).copy()
dataFrame.head()
for i in range(20):
    dataFrame.loc[i].plot(figsize=(10, 5))
lag0 = dataFrame.T.copy()
lag1 = lag0.shift(1, axis=0)
lag1.fillna(0, inplace=True)
lag1.index = lag1.index.astype(str) + '_lag1'
lag2 = lag0.shift(2, axis=0)
lag2.fillna(0, inplace=True)
lag2.index = lag2.index.astype(str) + '_lag2'
lag3 = lag0.shift(3, axis=0)
lag3.fillna(0, inplace=True)
lag3.index = lag3.index.astype(str) + '_lag3'
lag4 = lag0.shift(4, axis=0)
lag4.fillna(0, inplace=True)
lag4.index = lag4.index.astype(str) + '_lag4'
lag5 = lag0.shift(5, axis=0)
lag5.fillna(0, inplace=True)
lag5.index = lag5.index.astype(str) + '_lag5'
lag6 = lag0.shift(6, axis=0)
lag6.fillna(0, inplace=True)
lag6.index = lag6.index.astype(str) + '_lag6'
lag7 = lag0.shift(7, axis=0)
lag7.fillna(0, inplace=True)
lag7.index = lag7.index.astype(str) + '_lag7'
df = pd.concat((lag0, lag1, lag2, lag3, lag4, lag5, lag6, lag7), axis=0).T
df.head()
(fig1, ax) = plt.subplots(1, 1, figsize=(8, 4), dpi=100)
time0_columns = df[['time0', 'time0_lag1', 'time0_lag2', 'time0_lag3', 'time0_lag4', 'time0_lag5', 'time0_lag6', 'time0_lag7']].copy()
time0_columns.plot(ax=ax)
ax.set_xlabel('ID')
ax.set_ylabel('item_cnt_month')
ax.set_title('time0 has no lags')
(fig1, ax) = plt.subplots(1, 1, figsize=(8, 4), dpi=100)
time1_columns = df[['time1', 'time1_lag1', 'time1_lag2', 'time1_lag3', 'time1_lag4', 'time1_lag5', 'time1_lag6', 'time1_lag7']].copy()
time1_columns.plot(ax=ax)
ax.set_xlabel('ID')
ax.set_ylabel('item_cnt_month')
ax.set_title('time1 and its lag')
(fig1, ax) = plt.subplots(1, 1, figsize=(8, 4), dpi=100)
time30_columns = df[['time30', 'time30_lag1', 'time30_lag2', 'time30_lag3', 'time30_lag4', 'time30_lag5', 'time30_lag6', 'time30_lag7']].copy()
time30_columns.plot(ax=ax)
ax.set_xlabel('ID')
ax.set_ylabel('item_cnt_month')
ax.set_title('time30 and its lag')
from sklearn.decomposition import PCA
(fig1, ax) = plt.subplots(1, 1, figsize=(4, 4), dpi=100)
pca = PCA(n_components=5)