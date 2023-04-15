import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
cats = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
train.head(3)
temp2 = train.groupby('date')['item_id'].count()
df = pd.DataFrame(temp2)
df.columns.map({'item_id': 'item_id_cnt'})
df.rename(columns={'item_id': 'item_id_cnt'}, inplace=True)
df.plot()
temp = train[train['date'] == '02.01.2013']
temp.head(3)
temp2 = train[train['item_id'] == 5037][train['shop_id'] == 5][['date', 'item_cnt_day']]
import datetime

def str_to_datetime(s):
    split = s.split('.')
    (d, m, y) = (int(split[0]), int(split[1]), int(split[2]))
    return datetime.datetime(year=y, month=m, day=d)
temp2['date'] = temp2['date'].apply(str_to_datetime).sort_values()
temp2
train.head(3)
import matplotlib.pyplot as plt
import seaborn as sns
(f, ax) = plt.subplots(figsize=(8, 6))
sns.boxplot(x=train['item_price'])
(f, ax) = plt.subplots(figsize=(8, 6))
sns.boxplot(x=train['item_cnt_day'])
train = train[(train.item_price < 250000) & (train.item_cnt_day < 1000)]
train = train[train.item_price > 0].reset_index(drop=True)
train.loc[train.item_cnt_day < 1, 'item_cnt_day'] = 0
shops['city'] = shops.shop_name.str.split(' ').map(lambda x: x[0])
shops['category'] = shops.shop_name.str.split(' ').map(lambda x: x[1])
shops.head(3)
category = []
for cat in shops['category'].unique():
    if len(shops[shops['category'] == cat]) >= 5:
        category.append(cat)
shops.category = shops.category.apply(lambda x: x if x in category else 'others')
shops.head(3)
from sklearn.preprocessing import LabelEncoder
shops['category'] = LabelEncoder().fit_transform(shops.category)
shops['city'] = LabelEncoder().fit_transform(shops.city)
shops = shops[['shop_id', 'city', 'category']]
cats['type'] = cats.item_category_name.apply(lambda x: x.split(' ')[0]).astype(str)
categ = []
for cat in cats.type.unique():
    if len(cats[cats.type == cat]) >= 5:
        categ.append(cat)
cats.type = cats.type.apply(lambda x: x if x in categ else 'others')
cats.head(3)
cats['type_code'] = LabelEncoder().fit_transform(cats.type)
cats['split'] = cats.item_category_name.apply(lambda x: x.split('-'))
cats['sub_type'] = cats.split.apply(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
cats['sub_type_code'] = LabelEncoder().fit_transform(cats.sub_type)
cats = cats[['item_category_id', 'type_code', 'sub_type_code']]
items.head(10)
items.drop(['item_name'], inplace=True, axis=1)
items.head(3)
train.head(3)
train.date_block_num.nunique()
from itertools import product
matrix = []
cols = ['date_block_num', 'shop_id', 'item_id']
for i in range(34):
    sales = train[train.date_block_num == i]
    matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique()))))
matrix = pd.DataFrame(np.vstack(matrix), columns=cols)
matrix.head(3)
matrix['date_block_num'] = matrix['date_block_num'].astype(int)
matrix['shop_id'] = matrix['shop_id'].astype(np.int8)
matrix['item_id'] = matrix['item_id'].astype(np.int16)
matrix.sort_values(cols, inplace=True)
train['revenue'] = train['item_cnt_day'] * train['item_price']
group = train.groupby(cols).agg({'item_cnt_day': ['sum']})
group.columns = ['item_cnt_month']
group.reset_index(inplace=True)
group
matrix = pd.merge(matrix, group, on=cols, how='left')
matrix['item_cnt_month'] = matrix['item_cnt_month'].fillna(0).astype(np.float16)
test['date_block_num'] = 34
test['date_block_num'] = test['date_block_num'].astype(np.int8)
test['shop_id'] = test['shop_id'].astype(np.int8)
test['item_id'] = test['item_id'].astype(np.int16)
test.drop('ID', inplace=True, axis=1)
test.shape
matrix = pd.concat([matrix, test], ignore_index=True, sort=False, keys=cols)
matrix.fillna(0, inplace=True)
matrix.tail(5)
matrix[matrix['date_block_num'] == 34].shape
shops.head(2)
matrix = pd.merge(matrix, shops, on=['shop_id'], how='left')
matrix.head(4)
items.head(2)
matrix = pd.merge(matrix, items, on='item_id', how='left')
matrix.head(3)
cats.head(2)
matrix = pd.merge(matrix, cats, on='item_category_id', how='left')
matrix.head(3)
matrix.tail(3)
matrix['city'] = matrix['city'].astype(np.int8)
matrix['category'] = matrix['category'].astype(np.int8)
matrix['item_category_id'] = matrix['item_category_id'].astype(np.int8)
matrix['sub_type_code'] = matrix['sub_type_code'].astype(np.int8)
matrix['type_code'] = matrix['type_code'].astype(np.int8)
matrix.head(3)
matrix[matrix['date_block_num'] == 34].shape
matrix.head(3)

def lag_feature(df, lags, cols):
    for col in cols:
        print('Adding lag feature in ', col)
        tmp = df[['date_block_num', 'shop_id', 'item_id', col]]
        for i in lags:
            shifted = tmp.copy()
            shifted.columns = ['date_block_num', 'shop_id', 'item_id', col + '_shifted_' + str(i)]
            shifted.date_block_num = shifted.date_block_num + i
            df = pd.merge(df, shifted, on=['date_block_num', 'shop_id', 'item_id'], how='left')
    return df
matrix = lag_feature(matrix, [1, 2], ['item_cnt_month'])
group = matrix.groupby(['date_block_num', 'item_category_id']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_item_cat_avg']
group.reset_index(inplace=True)
matrix = pd.merge(matrix, group, on=['date_block_num', 'item_category_id'], how='left')
matrix['date_item_cat_avg'] = matrix['date_item_cat_avg'].astype(np.float16)
matrix = lag_feature(matrix, [1, 2], ['date_item_cat_avg'])
matrix.drop(['date_item_cat_avg'], axis=1, inplace=True)
group = matrix.groupby(['date_block_num', 'category']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_cat_avg']
group.reset_index(inplace=True)
matrix = pd.merge(matrix, group, on=['date_block_num', 'category'], how='left')
matrix['date_cat_avg'] = matrix['date_cat_avg'].astype(np.float16)
matrix = lag_feature(matrix, [1, 2], ['date_cat_avg'])
matrix.drop(['date_cat_avg'], axis=1, inplace=True)
group = matrix.groupby(['date_block_num']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_avg_item_cnt']
group.reset_index(inplace=True)
matrix = pd.merge(matrix, group, on='date_block_num', how='left')
matrix.date_avg_item_cnt = matrix['date_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1, 2], ['date_avg_item_cnt'])
matrix.drop(['date_avg_item_cnt'], inplace=True, axis=1)
group = matrix.groupby(['date_block_num', 'item_id']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_item_avg_item_cnt']
group.reset_index(inplace=True)
matrix = pd.merge(matrix, group, on=['date_block_num', 'item_id'], how='left')
matrix.date_item_avg_item_cnt = matrix['date_item_avg_item_cnt'].astype(np.float16)
matrix.head(3)
matrix = lag_feature(matrix, [1, 2], ['date_item_avg_item_cnt'])
matrix.drop(['date_item_avg_item_cnt'], inplace=True, axis=1)
group = train.groupby(['item_id']).agg({'item_price': ['mean']})
group.columns = ['item_id_price_avg']
group.reset_index(inplace=True)
matrix = pd.merge(matrix, group, on=['item_id'], how='left')
matrix['item_id_price_avg'] = matrix['item_id_price_avg'].astype(np.float16)
group = train.groupby(['date_block_num', 'item_id']).agg({'item_price': ['mean']})
group.columns = ['date_item_id_price_avg']
group.reset_index(inplace=True)
matrix = pd.merge(matrix, group, on=['date_block_num', 'item_id'], how='left')
matrix['date_item_id_price_avg'] = matrix['date_item_id_price_avg'].astype(np.float16)
matrix = lag_feature(matrix, [1, 2], ['date_item_id_price_avg'])
matrix.head(2)
for i in [1, 2]:
    matrix['delta_price_shifted_' + str(i)] = (matrix['date_item_id_price_avg_shifted_' + str(i)] - matrix['item_id_price_avg']) / matrix['item_id_price_avg']
features_to_drop = ['item_id_price_avg', 'date_item_id_price_avg']
matrix.drop(features_to_drop, axis=1, inplace=True)
matrix = matrix[matrix['date_block_num'] > 2]
from xgboost import XGBRegressor
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = (12, 4)
X_train = matrix[matrix.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = matrix[matrix.date_block_num < 33]['item_cnt_month']
X_valid = matrix[matrix.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = matrix[matrix.date_block_num == 33]['item_cnt_month']
X_test = matrix[matrix.date_block_num == 34].drop(['item_cnt_month'], axis=1)
Y_train = Y_train.clip(0, 20)
Y_valid = Y_valid.clip(0, 20)
model = XGBRegressor(max_depth=10, n_estimators=1000, min_child_weight=0.5, colsample_bytree=0.8, subsample=0.8, eta=0.1, seed=42)