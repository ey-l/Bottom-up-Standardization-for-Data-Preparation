import numpy as np
import pandas as pd
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
print(train.shape)
train.head()
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
test.head()
sample = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
sample.head()
train.head()
train.info()
train['date'][0]
train['date'][0][:2]
train['date'].apply(lambda x: x[:2])
train['day'] = train['date'].apply(lambda x: int(x[:2]))
train.head()
train['day'].describe()
train['date'][0][3:5]
train['month'] = train['date'].apply(lambda x: int(x[3:5]))
train['month'].describe()
train['year'] = train['date'].apply(lambda x: int(x[6:]))
train['year'].describe()
train.head()
train.drop(columns='date', inplace=True)
train.head()
print(train['year'].max())
train_2015 = train.loc[train['year'] == 2015]
print(train_2015.shape)
print(train_2015['month'].max())
print(train_2015.loc[train_2015['month'] == 10]['day'].max())
test.head()
test['year'] = 2015
test['month'] = 11
test.head()
train['date_block_num'].describe()
train.loc[train['date_block_num'] == 0]
train.loc[train['date_block_num'] == 1]
train.loc[train['date_block_num'] == 33]
test['date_block_num'] = 34
test.head()
train.loc[(train['shop_id'] == 0) & (train['item_id'] == 32)]
train.loc[(train['shop_id'] == 59) & (train['item_id'] == 22162)]
shops = train['shop_id'].unique()
items = train['item_id'].unique()
from tqdm.notebook import trange
all_data = []
for i in trange(34):
    for shop in shops:
        for item in items:
            all_data.append([i, shop, item])
all_data = pd.DataFrame(all_data, columns=['date_block_num', 'shop_id', 'item_id'])
print(all_data.shape)
print(train.shape)
train.groupby(['date_block_num', 'shop_id', 'item_id'], as_index=False)['item_cnt_day'].sum()
train.loc[(train['date_block_num'] == 0) & (train['shop_id'] == 0) & (train['item_id'] == 32)]
all_data = all_data.merge(train.groupby(['date_block_num', 'shop_id', 'item_id'], as_index=False)['item_cnt_day'].sum(), on=['date_block_num', 'shop_id', 'item_id'], how='left')
all_data.head()
all_data.rename(columns={'item_cnt_day': 'item_cnt_month'}, inplace=True)
all_data.fillna(0, inplace=True)
all_data.head()
all_data['year'] = 0
all_data['month'] = 0
year = 2013
month = 1
for date_block_num in range(34):
    all_data.loc[all_data['date_block_num'] == date_block_num, 'year'] = year
    all_data.loc[all_data['date_block_num'] == date_block_num, 'month'] = month
    month += 1
    if month == 13:
        year += 1
        month = 1
all_data.head()
all_data.tail()
all_data.head()
test.head()
import gc
train = all_data.loc[all_data['date_block_num'] != 33]
valid = all_data.loc[all_data['date_block_num'] == 33]
del all_data
gc.collect()
import lightgbm as lgb
params = {'objective': 'regression', 'metric': 'rmse', 'verbosity': -1}
use_cols = ['date_block_num', 'shop_id', 'item_id', 'year', 'month']
target = 'item_cnt_month'
X_train = train[use_cols]
y_train = train[target]
X_valid = valid[use_cols]
y_valid = valid[target]
train_set = lgb.Dataset(X_train, y_train)
valid_set = lgb.Dataset(X_valid, y_valid)
del train, valid
gc.collect()
model = lgb.train(params=params, train_set=train_set, valid_sets=[valid_set], num_boost_round=1000, early_stopping_rounds=10)
X_test = test[use_cols]
preds = model.predict(X_test)
submit = pd.DataFrame()
submit['ID'] = test['ID']
submit[target] = preds
submit.head()
print(submit.shape, sample.shape)
