import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_data = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
train_data.head()
train_data = train_data[train_data.item_cnt_day > 0]
train_data = train_data[train_data.item_price > 0]
train_data.info()
train_data.isnull().sum()
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
items.head()
item_cats = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
item_cats.head()
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
shops.head()
test_data = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
test_data.head()
train_data['date'] = pd.to_datetime(train_data['date'], format='%d.%m.%Y')
train_data['date']
train_data = train_data.join(items.set_index('item_id'), on='item_id')
train_data = train_data.join(item_cats.set_index('item_category_id'), on='item_category_id')
train_data = train_data.join(shops.set_index('shop_id'), on='shop_id')
import seaborn as sns
sns.scatterplot(data=train_data, x='item_price', y='item_cnt_day')
train_data.loc[train_data.item_price > 300000]
train_data.loc[train_data.item_cnt_day > 900]
train_data = train_data.drop(labels=[2326930, 2909818, 1163158], axis=0)
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 10))
sns.barplot(data=train_data, x='date_block_num', y='item_cnt_day')
from sklearn.preprocessing import LabelEncoder
lbl_encoder = LabelEncoder()
train_data['item_category_name'] = lbl_encoder.fit_transform(train_data['item_category_name'])
train_data['shop_name'] = lbl_encoder.fit_transform(train_data['shop_name'])
train_data = train_data.drop(['date', 'item_name'], axis=1)
train_data
monthly_train_data = train_data.groupby(by=['date_block_num', 'shop_id', 'item_id'], as_index=False).item_cnt_day.sum()
monthly_train_data.tail()
monthly_train_data = monthly_train_data.rename(columns={'item_cnt_day': 'item_cnt_month'})
monthly_train_data = monthly_train_data.join(items.set_index('item_id'), on='item_id')
monthly_train_data = monthly_train_data.join(item_cats.set_index('item_category_id'), on='item_category_id')
monthly_train_data = monthly_train_data.join(shops.set_index('shop_id'), on='shop_id')
monthly_train_data.head()
test_data.head()
test_data['date_block_num'] = 34
test_data.head()
test_data = test_data.join(items.set_index('item_id'), on='item_id')
test_data = test_data.join(item_cats.set_index('item_category_id'), on='item_category_id')
test_data = test_data.join(shops.set_index('shop_id'), on='shop_id')
test_data = test_data.drop('item_name', axis=1)
test_data.head()
df_large = pd.concat([monthly_train_data, test_data])
lbl_encoder = LabelEncoder()
df_large['item_category_name'] = lbl_encoder.fit_transform(df_large['item_category_name'])
df_large['shop_name'] = lbl_encoder.fit_transform(df_large['shop_name'])
df_large = df_large.drop('item_name', axis=1)
df_large.tail()
monthly_train_data.tail()
train_data = df_large.loc[df_large.date_block_num < 34]
train_data = train_data.drop('ID', axis=1)
train_data.tail()
test_data = df_large.loc[df_large.date_block_num == 34]
test_data = test_data.drop('item_cnt_month', axis=1)
test_data.head()
"\nfrom sklearn.feature_selection import mutual_info_regression\ny = train_data['item_cnt_month']\nX = train_data.drop('item_cnt_month', axis = 1)\ndisc_feat = X.dtypes == int\nmi_scores = mutual_info_regression(X, y, discrete_features=disc_feat, random_state = 10)\nmi_scores = pd.Series(mi_scores, name = 'MI scores', index = X.columns)\nmi_scores = mi_scores.sort_values(ascending = False)\nmi_scores\n"
'\nitem_id               0.229333\nitem_category_id      0.062293\nitem_category_name    0.062293\nshop_id               0.013078\nshop_name             0.013078\ndate_block_num        0.003998\nName: MI scores, dtype: float64\n'
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
y = train_data['item_cnt_month']
X = train_data.drop('item_cnt_month', axis=1)
(x_train, x_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=10)
rf = RandomForestRegressor()