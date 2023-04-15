import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df_train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
df_train.head(5)
df_test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
df_test.head(5)
df_items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
df_items.head(5)
df_train.info()
(f, ax) = plt.subplots(figsize=(18, 18))
sns.heatmap(df_train.corr(), annot=True, linewidths=0.5, fmt='.1f', ax=ax)

df_train.columns
df_test.columns
df_test['date_block_num'] = 34
df_test = df_test[['date_block_num', 'shop_id', 'item_id']]
df_test.head(5)
item_price = dict(df_train.groupby('item_id')['item_price'].last().reset_index().values)
df_test['item_price'] = df_test.item_id.map(item_price)
df_test.head(5)
df_train = df_train[df_train.item_id.isin(df_test.item_id)]
df_train = df_train[df_train.shop_id.isin(df_test.shop_id)]
dictionary = {'spain': 'madrid', 'usa': 'vegas'}
print(dictionary.keys())
print(dictionary.values())
dictionary['spain'] = 'barcelona'
print(dictionary)
dictionary['france'] = 'paris'
print(dictionary)
del dictionary['spain']
print(dictionary)
print('france' in dictionary)
df_train = df_train.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_price': 'last', 'item_cnt_day': 'sum'}).reset_index()
df_train.head(5)
df_train['shop*item'] = df_train.shop_id * df_train.item_id
df_train.head(5)
df_test['shop*item'] = df_test.shop_id * df_test.item_id
df_test.head(5)
df_items.drop('item_name', axis=1, inplace=True)
item_cat = dict(df_items.values)
df_train['item_cat'] = df_train.item_id.map(item_cat)
df_train.head(5)
df_test['item_cat'] = df_test.item_id.map(item_cat)
df_test.head(5)
df = pd.concat([df_train, df_test])
df.item_price = np.log1p(df.item_price)
df.item_price = df.item_price.fillna(df.item_price.mean())
df.item_cnt_day = df.item_cnt_day.apply(lambda x: 10 if x > 10 else x)
df.head(5)

def encode_the_numbers(column):
    helper_df = df.groupby(column)['item_cnt_day'].mean().sort_values(ascending=False).reset_index().reset_index()
    maper = helper_df.groupby(column)['index'].mean().to_dict()
    df[f'{column}_mean'] = df[column].map(maper)
    helper_df = df.groupby(column)['item_cnt_day'].sum().sort_values(ascending=False).reset_index().reset_index()
    maper = helper_df.groupby(column)['index'].sum().to_dict()
    df[f'{column}_sum'] = df[column].map(maper)
    helper_df = df.groupby(column)['item_cnt_day'].count().sort_values(ascending=False).reset_index().reset_index()
    maper = helper_df.groupby(column)['index'].count().to_dict()
    df[f'{column}_count'] = df[column].map(maper)
columns_to_encode = ['shop_id', 'item_id', 'shop*item', 'item_cat']
for column in columns_to_encode:
    encode_the_numbers(column)
corr_df = df.select_dtypes('number').drop('item_cnt_day', axis=1).corrwith(df.item_cnt_day).sort_values().reset_index().rename(columns={'index': 'feature', 0: 'correlation'})
(fig, ax) = plt.subplots(figsize=(5, 20))
ax.barh(y=corr_df.feature, width=corr_df.correlation)
ax.set_title('correlation between featuer and target'.title(), fontsize=16, fontfamily='serif', fontweight='bold')

df_train = df[df.item_cnt_day.notnull()]
df_train.head(5)
df_test = df[df.item_cnt_day.isnull()]
df_test.drop('item_cnt_day', axis=1, inplace=True)
df_test.head(5)
X = df_train.drop('item_cnt_day', axis=1).values
y = df_train.item_cnt_day.values
SC = MinMaxScaler()