import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
item_categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
sample_submission = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
print('item_categories')

print('items')

print('shops')

print('train')

print('test')

print('sample_submission')

train.info()
print('train')

print('test')

print('train')

print('test')

subset = ['date', 'date_block_num', 'shop_id', 'item_id', 'item_cnt_day']
print(train.duplicated(subset=subset).value_counts())
train.drop_duplicates(subset=subset, inplace=True)
train[train['item_price'] < 0]
train = train[train['item_price'] > 0]
train = train[train['item_cnt_day'] > 0]
sns.boxplot(train['item_price'])
sns.boxplot(train['item_cnt_day'])

def drop_outliers(df, feature, percentile_high=0.99):
    shape_init = df.shape[0]
    max_value = df[feature].quantile(percentile_high)
    print('dropping outliers...')
    df = df[df[feature] < max_value]
    print(str(shape_init - df.shape[0]) + ' ' + feature + ' values over ' + str(max_value) + ' have been removed')
    return df
train = drop_outliers(train, 'item_price')
train = drop_outliers(train, 'item_cnt_day')
prices_shop_df = train[['shop_id', 'item_id', 'item_price']]
prices_shop_df = prices_shop_df.groupby(['shop_id', 'item_id']).apply(lambda df: df['item_price'][-2:].mean())
prices_shop_df = prices_shop_df.to_frame(name='item_price')
prices_shop_df
test = pd.merge(test, prices_shop_df, how='left', left_on=['shop_id', 'item_id'], right_on=['shop_id', 'item_id'])
test.head()
test['item_price'].isnull().sum()
train['month'] = [date.split('.')[1] for date in train['date']]
train['year'] = [date.split('.')[2] for date in train['date']]
train.drop(['date', 'date_block_num'], axis=1, inplace=True)
test['month'] = '11'
test['year'] = '2015'
train_monthly = train.groupby(['year', 'month', 'shop_id', 'item_id'], as_index=False)[['item_cnt_day']].sum()
train_monthly.rename(columns={'item_cnt_day': 'item_cnt_month'}, inplace=True)
train_monthly = pd.merge(train_monthly, prices_shop_df, how='left', left_on=['shop_id', 'item_id'], right_on=['shop_id', 'item_id'])
train_monthly.head()
train = train_monthly
test = test.reindex(columns=['ID', 'year', 'month', 'shop_id', 'item_id', 'item_price'])
test.head()
item_categories['main_category'] = [x.split(' - ')[0] for x in item_categories['item_category_name']]
sub_categories = []
for i in range(len(item_categories)):
    try:
        sub_categories.append(item_categories['item_category_name'][i].split(' - ')[1])
    except IndexError as e:
        sub_categories.append('None')
item_categories['sub_category'] = sub_categories
item_categories.drop(['item_category_name'], axis=1, inplace=True)
item_categories.head()
items = pd.merge(items, item_categories, how='left')
items.drop(['item_name', 'item_category_id'], axis=1, inplace=True)
items.head()
train = pd.merge(train, items, how='left')
test = pd.merge(test, items, how='left')
from string import punctuation
shops['shop_name_cleaned'] = shops['shop_name'].apply(lambda s: ''.join([x for x in s if x not in punctuation]))
shops['shop_city'] = shops['shop_name_cleaned'].apply(lambda s: s.split()[0])
shops['shop_type'] = shops['shop_name_cleaned'].apply(lambda s: s.split()[1])
shops['shop_name'] = shops['shop_name_cleaned'].apply(lambda s: ' '.join(s.split()[2:]))
shops.drop(['shop_name_cleaned'], axis=1, inplace=True)
shops.head()
train = pd.merge(train, shops, how='left')
test = pd.merge(test, shops, how='left')
print('train')

print('test')

test['item_price'] = test.groupby(['main_category', 'sub_category'])['item_price'].apply(lambda df: df.fillna(df.median()))
test['item_price'].isnull().sum()
test['item_price'] = test.groupby(['sub_category'])['item_price'].apply(lambda df: df.fillna(df.median()))
test['item_price'].isnull().sum()
test[test['item_price'].isnull()]
filler = train[(train['main_category'] == 'PC') & (train['sub_category'] == 'Гарнитуры/Наушники')]['item_price'].median()
test['item_price'].fillna(filler, inplace=True)
test['item_price'].isnull().sum()
train['item_cnt_month'] = train['item_cnt_month'].clip(0, 20)
target_array = train['item_cnt_month']
train.drop(['item_cnt_month'], axis=1, inplace=True)
test_id = test['ID']
test.drop(['ID'], axis=1, inplace=True)
train.drop(['shop_id', 'item_id'], axis=1, inplace=True)
test.drop(['shop_id', 'item_id'], axis=1, inplace=True)

def downcast_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == 'float64']
    int_cols = [c for c in df if df[c].dtype == 'int64']
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int32)
    return df
downcast_dtypes(train)
downcast_dtypes(test)
train.info()
print('missing data in the train dataset : ', train.isnull().any().sum())
print('missing data in the test dataset : ', test.isnull().any().sum())

def normalityTest(data, alpha=0.05):
    from scipy import stats
    (statistic, p_value) = stats.normaltest(data)
    if p_value < alpha:
        is_normal_dist = False
    else:
        is_normal_dist = True
    return is_normal_dist
for feature in train.columns:
    if train[feature].dtype != 'object':
        if normalityTest(train[feature]) == False:
            train[feature] = np.log1p(train[feature])
            test[feature] = np.log1p(test[feature])
target_array = np.log1p(target_array)
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
X = enc.fit_transform(train)
y = target_array
X_predict = enc.fit_transform(test)
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.1, random_state=0)
from xgboost import XGBRegressor
model = XGBRegressor()