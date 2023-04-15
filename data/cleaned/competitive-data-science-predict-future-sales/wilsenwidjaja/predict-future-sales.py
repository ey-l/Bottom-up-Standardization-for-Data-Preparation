import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR as sk_SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
train.info()
item_category = items.groupby('item_category_id')['item_id'].count().reset_index()
(fig, ax) = plt.subplots(figsize=(20, 4))
sns.barplot(x=item_category.item_category_id, y=item_category.item_id, color='mediumblue')
ax.set(xlabel='Item Category ID', ylabel='Number of Item', title='Total Item Per Category')
sns.despine()
sns.boxplot(x=train['item_price'])
sns.boxplot(x=train['item_cnt_day'])
train['shop_id'] = train['shop_id'].replace({0: 57, 1: 58, 11: 10, 40: 39})
train = train.loc[train.shop_id.isin(test['shop_id'].unique()), :]
train = train[(train['item_price'] > 0) & (train['item_price'] < 45000)]
train = train[(train['item_cnt_day'] > 0) & (train['item_cnt_day'] < 800)]
item_categories = []
for i in train['item_id']:
    item_categories.append(items['item_category_id'].iloc[i])
train['item_categories'] = item_categories
train.info()
shop_item = train.groupby('shop_id')['item_cnt_day'].sum().reset_index()
(fig, ax) = plt.subplots(figsize=(20, 4))
sns.barplot(x=shop_item.shop_id, y=shop_item.item_cnt_day, color='mediumblue')
ax.set(xlabel='Shop ID', ylabel='Number of Sales', title='Total Sales Per Shop')
sns.despine()

def monthly_sales(data):
    data = data.copy()
    data.date = data.date.apply(lambda x: str(x)[3:])
    data = data.groupby(['date', 'date_block_num', 'shop_id', 'item_id', 'item_price', 'item_categories'])['item_cnt_day'].sum().reset_index()
    data['date'] = data['date'].apply(lambda x: x.replace('.', '-'))
    data['date'] = pd.to_datetime(data['date'])
    return data
monthly_data = monthly_sales(train)

def time_plot(data, x_col, y_col, title):
    (fig, ax) = plt.subplots(figsize=(20, 5))
    sns.lineplot(x=x_col, y=y_col, data=data, ax=ax, color='mediumblue', label='Total Sales')
    second = data.groupby(data.date.dt.year)[y_col].mean().reset_index()
    second.date = pd.to_datetime(second.date, format='%Y')
    sns.lineplot(x=second.date + datetime.timedelta(6 * 365 / 12), y=y_col, data=second, ax=ax, color='red', label='Mean Sales')
    ax.set(xlabel='Date', ylabel='Sales', title=title)
    sns.despine()
time_plot(monthly_data, 'date', 'item_cnt_day', 'Sales Trend')
train.info()
dl_train = train.drop(['date', 'date_block_num', 'item_cnt_day', 'item_price'], axis=1)
dl_target = train['item_cnt_day']
dl_train = np.array(dl_train)
dl_target = np.array(dl_target)
from sklearn.preprocessing import StandardScaler
dl_train = StandardScaler().fit_transform(dl_train)
model = Sequential()
model.add(Dense(16, activation='LeakyReLU', input_shape=(dl_train.shape[1],)))
model.add(Dropout(rate=0.2))
model.add(Dense(8, activation='sigmoid'))
model.add(Dense(4, activation='sigmoid'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mse', 'accuracy'])
model.summary()