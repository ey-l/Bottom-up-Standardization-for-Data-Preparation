import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import seaborn as sns
import matplotlib

from matplotlib import pyplot as plt
import numpy as np
pd.set_option('display.max_rows', 50)
sns.set(rc={'figure.figsize': (11, 4)})
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import warnings
warnings.filterwarnings('ignore')
item_category = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv', parse_dates=['date'])
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
print(item_category.isnull().sum())
print(items.isnull().sum())
print(train.isnull().sum())
print(shops.isnull().sum())
print(test.isnull().sum())
print('No null items in all files')
print('item_category columns name', item_category.columns)
print('items columns name', items.columns)
print('train columns name', train.columns)
print('shops', shops.columns)
print('test', test.columns)
df = pd.merge(train, items, on='item_id', how='left')
print(df)
df.describe()
df.isnull().sum()
df.info()
for i in df.columns:
    print(i, ' ', df[i].nunique())
for i in df.columns:
    print(i, '', df[i].unique())
df.describe(include='all')
df['item_price'].hist()
df[df['item_price'] > 40000].count()
sns.boxplot(x=df['item_price'], data=df)
sns.displot(df, x='item_price', kind='kde')
sns.boxplot(df['item_cnt_day'])
df[df['item_cnt_day'] > 150].count()
sns.displot(df, x='item_cnt_day', kind='kde')
sns.scatterplot(x='item_price', y='item_cnt_day', data=df)
corr = df.corr()
corr
print(corr['item_cnt_day'].sort_values(ascending=False))
df['month'] = df['date'].dt.month
df.drop('date', axis=1, inplace=True)
df.drop(df.loc[df['item_price'] > 40000].index, inplace=True)
df.shape
df.drop('item_name', axis=1, inplace=True)
df.drop(df.loc[df['item_cnt_day'] > 150].index, inplace=True)
df_2 = df.groupby(['date_block_num', 'shop_id', 'item_id', 'item_category_id', 'month']).agg({'item_price': 'mean', 'item_cnt_day': 'sum'}).reset_index()
df_2 = df.rename(columns={'item_cnt_day': 'item_cnt_month'}, inplace=False)
test['month'] = int('11')
test['date_block_num'] = 34
test.head()
df_3 = df.groupby(['shop_id', 'item_id'])['item_price'].last().reset_index()
test = pd.merge(test, df_3, on=['shop_id', 'item_id'], how='left')
print(test)
sns.displot(test, x='item_price', kind='kde')
print(test.isnull().sum())
test['item_price'] = test['item_price'].fillna(test['item_price'].median())
test['item_price']
test = pd.merge(test, items, on=['item_id'], how='left')
test.head()
test.drop('item_name', axis=1, inplace=True)
test.columns
test.isnull().sum()
test_X = test[['shop_id', 'item_id', 'month', 'date_block_num', 'item_price', 'item_category_id']]
y = df_2[['item_cnt_month']]
x = df_2.drop(['item_cnt_month'], axis=1)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()