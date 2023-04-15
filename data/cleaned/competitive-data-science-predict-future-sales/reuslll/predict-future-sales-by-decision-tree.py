import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
train_set = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
validation_set = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
items_data = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
item_categories_data = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
shops_data = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
train_set.head(10)
train_set.isnull().sum()
ax1 = plt.subplot(211)
plt.ylim(train_set.item_cnt_day.min(), train_set.item_cnt_day.max() * 1.2)
ax1.boxplot(x=train_set.item_cnt_day)
plt.xlabel('item_cnt_day')
ax2 = plt.subplot(212)
plt.ylim(train_set.item_price.min(), train_set.item_price.max() * 1.2)
ax2.boxplot(x=train_set.item_price)
plt.xlabel('item_price')
train_set = train_set[train_set.item_price < 100000]
train_set = train_set[(train_set.item_cnt_day < 1001) & (train_set.item_cnt_day >= 0)]
combine = []
for i in range(34):
    sales = train_set[train_set.date_block_num == i]
    for j in sales.shop_id.unique():
        for k in sales.item_id.unique():
            p = (i, j, k)
            combine.append(np.array(list(p)))
cols = ['date_block_num', 'shop_id', 'item_id']
combine = pd.DataFrame(np.vstack(combine), columns=cols)
grouped = train_set.groupby(['item_id', 'shop_id', 'date_block_num']).agg({'item_cnt_day': 'sum'})
grouped.columns = ['item_cnt_month']
grouped.reset_index(inplace=True)
grouped.head()
combine = combine.merge(grouped, on=['item_id', 'shop_id', 'date_block_num'], how='left')
combine['item_cnt_month'] = combine['item_cnt_month'].fillna(0).clip(0, 20)
combine.head()
combine = pd.merge(combine, items_data, on=['item_id'], how='left')

def ItemCatSplit(x):
    if '-' in x:
        cat = x.split(' - ')[0]
    else:
        cat = x
    return cat
item_categories_data['Cat'] = [ItemCatSplit(i) for i in item_categories_data['item_category_name']]

def ShopNameSplit(x):
    Provice = x.split(' ')[0]
    return Provice
shops_data['Location'] = [ShopNameSplit(i) for i in shops_data['shop_name']]
train = pd.merge(combine, item_categories_data, how='left', on='item_category_id')
train = pd.merge(train, shops_data, how='left', on='shop_id')
train = train.drop(columns=['item_category_name', 'item_name', 'shop_name'])
train.head()
col = list(train.columns)
for i in col:
    print('特征%s的种类有%d种' % (i, len(train[i].unique())))
test = pd.merge(validation_set, items_data, on='item_id', how='left')
test = pd.merge(test, item_categories_data, how='left', on='item_category_id')
test = pd.merge(test, shops_data, how='left', on='shop_id')
test = test.drop(columns=['item_name', 'item_category_name', 'shop_name'])
test['date_block_num'] = 34
test.head()
train_test = pd.concat([train, test])
train_test.isnull().sum()
oer = OrdinalEncoder()
oe = oer.fit_transform(train_test.iloc[:, 5:6])
train_test['Cat'] = oe
oe = oer.fit_transform(train_test.iloc[:, 6:7])
train_test['Location'] = oe
grouped = train_test.groupby(['shop_id', 'date_block_num']).agg({'item_cnt_month': 'mean'})
grouped.columns = ['shop_last_month_mean']
grouped.reset_index(inplace=True)
grouped['date_block_num'] = grouped['date_block_num'] + 1
train_test = train_test.merge(grouped, on=['shop_id', 'date_block_num'], how='left')
train_test['shop_last_month_mean'] = train_test['shop_last_month_mean'].fillna(0)
grouped = train_test.groupby(['item_id', 'date_block_num']).agg({'item_cnt_month': 'mean'})
grouped.columns = ['item_last_month_mean']
grouped.reset_index(inplace=True)
grouped['date_block_num'] = grouped['date_block_num'] + 1
train_test = train_test.merge(grouped, on=['item_id', 'date_block_num'], how='left')
train_test['item_last_month_mean'] = train_test['item_last_month_mean'].fillna(0)
grouped = train_test.groupby(['item_category_id', 'date_block_num']).agg({'item_cnt_month': 'mean'})
grouped.columns = ['category_last_month_mean']
grouped.reset_index(inplace=True)
grouped['date_block_num'] = grouped['date_block_num'] + 1
train_test = train_test.merge(grouped, on=['item_category_id', 'date_block_num'], how='left')
train_test['category_last_month_mean'] = train_test['category_last_month_mean'].fillna(0)
grouped = train_test.groupby(['shop_id', 'date_block_num']).agg({'item_cnt_month': 'mean'})
grouped.columns = ['shop_3month_ago_mean']
grouped.reset_index(inplace=True)
grouped['date_block_num'] = grouped['date_block_num'] + 3
train_test = train_test.merge(grouped, on=['shop_id', 'date_block_num'], how='left')
train_test['shop_3month_ago_mean'] = train_test['shop_3month_ago_mean'].fillna(0)
grouped = train_test.groupby(['item_id', 'date_block_num']).agg({'item_cnt_month': 'mean'})
grouped.columns = ['item_3month_ago_mean']
grouped.reset_index(inplace=True)
grouped['date_block_num'] = grouped['date_block_num'] + 3
train_test = train_test.merge(grouped, on=['item_id', 'date_block_num'], how='left')
train_test['item_3month_ago_mean'] = train_test['item_3month_ago_mean'].fillna(0)
grouped = train_test.groupby(['item_category_id', 'date_block_num']).agg({'item_cnt_month': 'mean'})
grouped.columns = ['category_3month_ago_mean']
grouped.reset_index(inplace=True)
grouped['date_block_num'] = grouped['date_block_num'] + 3
train_test = train_test.merge(grouped, on=['item_category_id', 'date_block_num'], how='left')
train_test['category_3month_ago_mean'] = train_test['category_3month_ago_mean'].fillna(0)
validation = train_test[train_test['date_block_num'] == 34]
train_x = train_test.query('date_block_num<33').drop(columns=['item_cnt_month', 'ID']).values
test_x = train_test[train_test['date_block_num'] == 33].drop(columns=['item_cnt_month', 'ID']).values
train_y = train_test.query('date_block_num<33')['item_cnt_month'].values
test_y = train_test[train_test['date_block_num'] == 33]['item_cnt_month'].values
rmse_list = []
for md in np.arange(5, 15):
    t = datetime.now()
    tree = DecisionTreeRegressor(max_depth=md, min_samples_leaf=5, random_state=42)