import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
from itertools import product
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

def plot_features(booster, figsize):
    (fig, ax) = plt.subplots(1, 1, figsize=figsize)
    return plot_importance(booster=booster, ax=ax)
item_categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
sales_train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
sample_submission = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
(sales_train.shape, sales_train.head())
(len(sales_train['item_id'].unique()), len(sales_train['shop_id'].unique()))
sales_train.describe()
plt.figure(figsize=(10, 4))
plt.xlim(-100, 3000)
sns.boxplot(x=sales_train.item_cnt_day)
sales_train = sales_train.loc[sales_train['item_cnt_day'] < 1000,].loc[sales_train['item_cnt_day'] >= 0,]
plt.figure(figsize=(10, 4))
plt.xlim(sales_train.item_price.min(), sales_train.item_price.max() * 1.1)
sns.boxplot(x=sales_train.item_price)
sales_train = sales_train.loc[sales_train['item_price'] < 90000,]
sales_train = sales_train.loc[sales_train['item_price'] > 0,]
sales_train.shape
sales_train.isna().sum()
spec_train = sales_train.sort_values('date').groupby(['date_block_num', 'shop_id'], as_index=False)
spec_train = spec_train.agg({'item_cnt_day': ['sum']})
spec_train.columns = ['date_block_num', 'shop_id', 'item_month_by_shop']
spec_train.head()
axis = np.arange('2013-01', '2015-11', dtype='datetime64[M]')
plt.figure(figsize=(20, 15))
list_of_shops = sorted(spec_train['shop_id'].unique())
for i in range(len(list_of_shops)):
    L = spec_train.loc[spec_train['shop_id'] == list_of_shops[i], 'item_month_by_shop']
    if L.shape[0] == 34:
        plt.plot(axis, spec_train.loc[spec_train['shop_id'] == list_of_shops[i], 'item_month_by_shop'])

sales_train['year'] = sales_train['date'].apply(lambda x: x[6:])
sales_train['year'] = sales_train['year'].astype('int64')
final_train = []
cols = ['date_block_num', 'shop_id', 'item_id']
for i in range(34):
    sales = sales_train[sales_train.date_block_num == i]
    final_train.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='int16'))
final_train = pd.DataFrame(np.vstack(final_train), columns=cols)
final_train['date_block_num'] = final_train['date_block_num'].astype(np.int8)
final_train['shop_id'] = final_train['shop_id'].astype(np.int8)
final_train['item_id'] = final_train['item_id'].astype(np.int16)
final_train.sort_values(cols, inplace=True)
dict_year = sales_train[['date_block_num', 'year']].set_index('date_block_num').to_dict()['year']
final_train['year'] = final_train['date_block_num'].map(dict_year)
train_monthly = sales_train.sort_values('date').groupby(['date_block_num', 'shop_id', 'item_id'], as_index=False)
train_monthly = train_monthly.agg({'item_cnt_day': ['sum']})
train_monthly.columns = ['date_block_num', 'shop_id', 'item_id', 'item_cnt_month']
train_monthly
final_train = pd.merge(final_train, train_monthly, on=cols, how='left')
final_train['item_cnt_month'] = final_train['item_cnt_month'].fillna(0).clip(0, 20).astype(np.float16)
final_train
final_train['month'] = final_train['date_block_num'] % 12
final_train1 = final_train.join(items, on='item_id', rsuffix='_').join(shops, on='shop_id', rsuffix='_').join(item_categories, on='item_category_id', rsuffix='_').drop(['item_id_', 'shop_id_', 'item_category_id_'], axis=1)
group = final_train1.groupby(['date_block_num']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_avg_item_cnt']
group.reset_index(inplace=True)
final_train1 = pd.merge(final_train1, group, on=['date_block_num'], how='left')
final_train1['date_avg_item_cnt'] = final_train1['date_avg_item_cnt'].astype(np.float16)
shifted = final_train1[['date_block_num', 'shop_id', 'item_id', 'date_avg_item_cnt']].copy()
shifted['date_avg_item_cnt_shift_1'] = shifted['date_avg_item_cnt']
shifted = shifted.drop(columns='date_avg_item_cnt')
shifted['date_block_num'] += 1
final_train1 = pd.merge(final_train1, shifted, on=['date_block_num', 'shop_id', 'item_id'], how='left').copy()
final_train1
group
test = test.drop(columns='ID')
test['date_block_num'] = 34
test['month'] = 11
test['year'] = 2015
test1 = test.join(items, on='item_id', rsuffix='_').join(shops, on='shop_id', rsuffix='_').join(item_categories, on='item_category_id', rsuffix='_').drop(['item_id_', 'shop_id_', 'item_category_id_'], axis=1)
test1['date_avg_item_cnt_shift_1'] = 0.259033
(test1.columns, final_train1.columns)
X_train = final_train1[final_train1.date_block_num < 32].drop(['item_cnt_month'], axis=1)[['date_block_num', 'shop_id', 'item_id', 'month', 'year', 'item_category_id', 'date_avg_item_cnt_shift_1']]
Y_train = final_train1[final_train1.date_block_num < 32]['item_cnt_month']
X_valid = final_train1[final_train1.date_block_num >= 32].drop(['item_cnt_month'], axis=1)[['date_block_num', 'shop_id', 'item_id', 'month', 'year', 'item_category_id', 'date_avg_item_cnt_shift_1']]
Y_valid = final_train1[final_train1.date_block_num >= 32]['item_cnt_month']
X_test = test1[['date_block_num', 'shop_id', 'item_id', 'month', 'year', 'item_category_id', 'date_avg_item_cnt_shift_1']]
model = XGBRegressor(max_depth=8, n_estimators=100, min_child_weight=30, colsample_bytree=0.8, tree_method='gpu_hist', subsample=0.8, eta=0.3, seed=42)