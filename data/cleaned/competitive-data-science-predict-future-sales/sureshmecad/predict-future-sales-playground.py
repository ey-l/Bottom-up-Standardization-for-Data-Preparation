import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('darkgrid')
import gc
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
item_cat = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
Shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
submission = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')






train = pd.merge(train, items, on='item_id', how='inner')
train = pd.merge(train, item_cat, on='item_category_id', how='inner')
train = pd.merge(train, Shops, on='shop_id', how='inner')
test = pd.merge(test, items, on='item_id', how='inner')
test = pd.merge(test, item_cat, on='item_category_id', how='inner')
test = pd.merge(test, Shops, on='shop_id', how='inner')










print('Number of duplicates in train:', len(train[train.duplicated()]))
print('Number of duplicates in test:', len(test[test.duplicated()]))
train.describe()
plt.figure(figsize=(8, 6))
sns.heatmap(train.corr(), annot=True, cbar=False, cmap='coolwarm')
items.head()
items.shape
items.dtypes
items.count()
print('Number of Duplicates in item:', len(items[items.duplicated()]))
print('Unique item names:', len(items['item_name'].unique()))
items.item_id.nunique()
items.item_category_id.nunique()
plt.figure(figsize=(18, 18))
items.groupby('item_category_id')['item_id'].size().plot.barh(rot=0)
plt.title('Number of items related to different categories')
plt.xlabel('Categories')
plt.ylabel('Number of items')
items.groupby('item_category_id')['item_id'].size().mean()
items.groupby('item_category_id')['item_id'].size().max()
items.groupby('item_category_id')['item_id'].size().min()
item_cat[item_cat['item_category_id'].isin(items.groupby('item_category_id')['item_id'].size().nlargest(5).index)]
item_cat[item_cat['item_category_id'].isin(items.groupby('item_category_id')['item_id'].size()[items.groupby('item_category_id')['item_id'].size() == 1].index)]
plt.figure(figsize=(10, 5))
sns.distplot(train['item_id'], color='red')
plt.figure(figsize=(10, 5))
sns.distplot(train['item_price'], color='red')
plt.figure(figsize=(10, 5))
sns.distplot(np.log(train['item_price']), color='red')
item_cat.head()
item_cat.shape
item_cat.dtypes
item_cat.count()
print('Number of Duplicates in item_cat:', len(item_cat[item_cat.duplicated()]))
print('Unique item names:', len(item_cat['item_category_id'].unique()))
item_cat['item_category_id'].nunique()
item_cat['item_category_id'].values
Shops.head()
Shops.shape
Shops.dtypes
Shops.count()
color = sns.color_palette('hls', 8)
sns.set(style='darkgrid')
plt.figure(figsize=(15, 5))
sns.countplot(x=train['shop_id'], data=train, palette=color)
train.isnull().sum()
test.isnull().sum()
items.isnull().sum()
item_cat.isnull().sum()
Shops.isnull().sum()
plt.figure(figsize=(8, 6))
sns.boxplot(x='item_price', data=train)
plt.hist(x='item_price')
train.item_cnt_day.plot()
plt.title('Number of products sold per day')
train.item_price.hist()
plt.title('Item Price Distribution')
sns.pairplot(train)

def Box_plots(df):
    plt.figure(figsize=(10, 4))
    plt.title('Box Plot')
    sns.boxplot(df)

Box_plots(train['item_price'])
Box_plots(train['item_cnt_day'])
sales = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv', parse_dates=['date'], infer_datetime_format=True, dayfirst=True)
sales.head()
df = sales.groupby([sales.date.apply(lambda x: x.strftime('%Y-%m')), 'item_id', 'shop_id']).sum().reset_index()
df = df[['date', 'item_id', 'shop_id', 'item_cnt_day']]
df = df.pivot_table(index=['item_id', 'shop_id'], columns='date', values='item_cnt_day', fill_value=0).reset_index()
df.head()
df_test = pd.merge(test, df, on=['item_id', 'shop_id'], how='left')
df_test = df_test.fillna(0)
df_test.head()
df_test = df_test.drop(labels=['ID', 'shop_id', 'item_id', 'item_name', 'item_category_name', 'shop_name'], axis=1)
df_test.head()
TARGET = '2015-10'
y_train = df_test[TARGET]
X_train = df_test.drop(labels=[TARGET], axis=1)
print(y_train.shape)
print(X_train.shape)
X_train.head()
print(y_train.shape)
print(X_train.shape)
X_test = df_test.drop(labels=['2013-01'], axis=1)
print(X_test.shape)
from lightgbm import LGBMRegressor
model = LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=32, colsample_bytree=0.9497036, subsample=0.8715623, max_depth=8, reg_alpha=0.04, reg_lambda=0.073, min_split_gain=0.0222415, min_child_weight=40)
print('Training time, it is...')