import pandas as pd
import numpy as np
data = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
data
train = data[data.date_block_num != 33]
train
test = data[data.date_block_num == 33]
drop_col = ['date', 'item_price', 'item_cnt_day']
test.drop(drop_col, axis=1, inplace=True)
test
print(train.shape)
print(test.shape)
train.head()
test.head()
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
print(shops.shape)
shops.head()
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
print(items.shape)
items.head()
cats = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
print(cats.shape)
cats.head()
sample_submission = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
print(sample_submission.shape)
sample_submission.head()
import matplotlib.pyplot as plt
import seaborn as sns
(fig, ax) = plt.subplots(2, 1, figsize=(10, 4))
plt.xlim(-300, 3000)
ax[0].boxplot(train.item_cnt_day, labels=['train.item_cnt_day'], vert=False)
plt.xlim(-1000, 350000)
ax[1].boxplot(train.item_price, labels=['train.item_price'], vert=False)

train = train[train.item_price < 100000]
train = train[train.item_cnt_day < 1001]
train[train.item_price < 0]
median = train[(train.date_block_num == 4) & (train.shop_id == 32) & (train.item_id == 2973) & (train.item_price > 0)].item_price.median()
train.loc[train.item_price < 0, 'item_price'] = median
train[train.item_price < 0]
shops
train.loc[train.shop_id == 0, 'shop_id'] = 57
test.loc[test.shop_id == 0, 'shop_id'] = 57
train.loc[train.shop_id == 1, 'shop_id'] = 58
test.loc[test.shop_id == 1, 'shop_id'] = 58
train.loc[train.shop_id == 10, 'shop_id'] = 11
test.loc[test.shop_id == 10, 'shop_id'] = 11
from sklearn.preprocessing import LabelEncoder
shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'
shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])
shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'
shops['city_code'] = LabelEncoder().fit_transform(shops['city'])
shops = shops[['shop_id', 'city_code']]
shops.head()
cats
cats['split'] = cats['item_category_name'].str.split('-')
cats['type'] = cats['split'].map(lambda x: x[0].strip())
cats['subtype'] = cats['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
cats['type_code'] = LabelEncoder().fit_transform(cats['type'])
cats['subtype_code'] = LabelEncoder().fit_transform(cats['subtype'])
cats = cats[['item_category_id', 'type_code', 'subtype_code']]
cats.head()
items
items.drop(['item_name'], axis=1, inplace=True)
items.head()
print(len(list(set(test.item_id) - set(test.item_id).intersection(set(train.item_id)))))
print(len(list(set(test.item_id))))
print(len(test))
train.date_block_num.unique()
from itertools import product
print(list(product([1], [1, 2, 3, 4])))
import time
from itertools import product
ts = time.time()
matrix = []
for i in range(33):
    sales = train[train.date_block_num == i]
    matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='int16'))
cols = ['date_block_num', 'shop_id', 'item_id']
matrix = pd.DataFrame(np.vstack(matrix), columns=cols)
matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)
matrix['shop_id'] = matrix['shop_id'].astype(np.int8)
matrix['item_id'] = matrix['item_id'].astype(np.int16)
matrix.sort_values(cols, inplace=True)
time.time() - ts
matrix.shape
train['revenue'] = train['item_price'] * train['item_cnt_day']
train.head()
group = train.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_day': ['sum']})
group.columns = ['item_cnt_month']
group.reset_index(inplace=True)
group.head()
matrix = pd.merge(matrix, group, on=cols, how='left')
matrix['item_cnt_month'] = matrix['item_cnt_month'].fillna(0).clip(0, 20).astype(np.float16)
test
test.dtypes
test['date_block_num'] = test['date_block_num'].astype(np.int8)
test['shop_id'] = test['shop_id'].astype(np.int8)
test['item_id'] = test['item_id'].astype(np.int16)
test.head()
ts = time.time()
matrix = pd.concat([matrix, test], ignore_index=True, sort=False, keys=cols)
matrix.fillna(0, inplace=True)
time.time() - ts
matrix.head()
ts = time.time()
matrix = pd.merge(matrix, shops, on=['shop_id'], how='left')
matrix = pd.merge(matrix, items, on=['item_id'], how='left')
matrix = pd.merge(matrix, cats, on=['item_category_id'], how='left')
matrix['city_code'] = matrix['city_code'].astype(np.int8)
matrix['item_category_id'] = matrix['item_category_id'].astype(np.int8)
matrix['type_code'] = matrix['type_code'].astype(np.int8)
matrix['subtype_code'] = matrix['subtype_code'].astype(np.int8)
time.time() - ts
matrix

def lag_feature(df, lags, col):
    tmp = df[['date_block_num', 'shop_id', 'item_id', col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num', 'shop_id', 'item_id', col + '_lag_' + str(i)]
        shifted['date_block_num'] += i
        df = pd.merge(df, shifted, on=['date_block_num', 'shop_id', 'item_id'], how='left')
    return df
matrix
ts = time.time()
matrix = lag_feature(matrix, [1, 2, 3, 6, 12], 'item_cnt_month')
time.time() - ts
matrix
import gc
import pickle
matrix.to_pickle('data.pkl')
gc.collect()
data = pd.read_pickle('data.pkl')
matrix.columns
data = data[matrix.columns]
data
X_train = data[data.date_block_num < 32].drop(['item_cnt_month'], axis=1)
Y_train = data[data.date_block_num < 32]['item_cnt_month']
X_valid = data[data.date_block_num == 32].drop(['item_cnt_month'], axis=1)
Y_valid = data[data.date_block_num == 32]['item_cnt_month']
X_test = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_true = data[data.date_block_num == 33]['item_cnt_month']
del data
gc.collect()
from xgboost import XGBRegressor
ts = time.time()
model = XGBRegressor(max_depth=8, n_estimators=1000, min_child_weight=300, colsample_bytree=0.8, subsample=0.8, eta=0.3, seed=42)