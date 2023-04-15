import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.float_format', lambda x: '%.3f' % x)

train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
submission = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
from pandas_profiling import ProfileReport
profile = ProfileReport(train, title='Perfil de reporte de ventas')
profile
train['item_id'].value_counts(ascending=False)[:5]
items.loc[items['item_id'] == 20949]
categories.loc[categories['item_category_id'] == 71]
test.loc[test['item_id'] == 20949].head(6)
items.loc[items['item_category_id'] == 71].head(20)
train['item_cnt_day'].sort_values(ascending=False)[:5]
train[train['item_cnt_day'] == 2169]
train[train['item_id'] == 11373].head(5)
train = train[train['item_cnt_day'] < 2000]
train['item_cnt_day'].sort_values(ascending=False)[:5]
train['item_price'].sort_values(ascending=False)[:5]
train[train['item_price'] == 307980]
items[items['item_id'] == 6066]
train[train['item_id'] == 6066]
train = train[train['item_price'] < 300000]
train['item_price'].sort_values()[:5]
train[train['item_price'] == -1]
train[train['item_id'] == 2973].head(5)
price_correction = train[(train['shop_id'] == 32) & (train['item_id'] == 2973) & (train['date_block_num'] == 4) & (train['item_price'] > 0)].item_price.median()
train.loc[train['item_price'] < 0, 'item_price'] = price_correction
shops_train = train['shop_id'].nunique()
shops_test = test['shop_id'].nunique()
print('Shops in Training Set: ', shops_train)
print('Shops in Test Set: ', shops_test)
shops_train_list = list(train['shop_id'].unique())
shops_test_list = list(test['shop_id'].unique())
flag = 0
if set(shops_test_list).issubset(set(shops_train_list)):
    flag = 1
if flag:
    print('Yes, list is subset of other.')
else:
    print('No, list is not subset of other.')
shops.T
train.loc[train['shop_id'] == 0, 'shop_id'] = 57
test.loc[test['shop_id'] == 0, 'shop_id'] = 57
train.loc[train['shop_id'] == 1, 'shop_id'] = 58
test.loc[test['shop_id'] == 1, 'shop_id'] = 58
train.loc[train['shop_id'] == 10, 'shop_id'] = 11
test.loc[test['shop_id'] == 10, 'shop_id'] = 11
cities = shops['shop_name'].str.split(' ').map(lambda row: row[0])
cities.unique()
shops['city'] = shops['shop_name'].str.split(' ').map(lambda row: row[0])
shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit_transform(shops['city'])
shops['city_label'] = le.fit_transform(shops['city'])
shops.drop(['shop_name', 'city'], axis=1, inplace=True)
shops.head()
items_train = train['item_id'].nunique()
items_test = test['item_id'].nunique()
print('Items in Training Set: ', items_train)
print('Items in Test Set: ', items_test)
items_train_list = list(train['item_id'].unique())
items_test_list = list(test['item_id'].unique())
flag = 0
if set(items_test_list).issubset(set(items_train_list)):
    flag = 1
if flag:
    print('Yes, list is subset of other.')
else:
    print('No, list is not subset of other.')
len(set(items_test_list).difference(items_train_list))
categories_in_test = items.loc[items['item_id'].isin(sorted(test['item_id'].unique()))].item_category_id.unique()
categories.loc[~categories['item_category_id'].isin(categories_in_test)].T
le = preprocessing.LabelEncoder()
main_categories = categories['item_category_name'].str.split('-')
categories['main_category_id'] = main_categories.map(lambda row: row[0].strip())
categories['main_category_id'] = le.fit_transform(categories['main_category_id'])
categories['sub_category_id'] = main_categories.map(lambda row: row[1].strip() if len(row) > 1 else row[0].strip())
categories['sub_category_id'] = le.fit_transform(categories['sub_category_id'])
categories.head()
train['date'] = pd.to_datetime(train['date'], format='%d.%m.%Y')
from itertools import product
shops_in_jan = train.loc[train['date_block_num'] == 0, 'shop_id'].unique()
items_in_jan = train.loc[train['date_block_num'] == 0, 'item_id'].unique()
jan = list(product(*[shops_in_jan, items_in_jan, [0]]))
print(len(jan))
shops_in_feb = train.loc[train['date_block_num'] == 1, 'shop_id'].unique()
items_in_feb = train.loc[train['date_block_num'] == 1, 'item_id'].unique()
feb = list(product(*[shops_in_feb, items_in_feb, [1]]))
print(len(feb))
cartesian_test = []
cartesian_test.append(np.array(jan))
cartesian_test.append(np.array(feb))
cartesian_test
cartesian_test = np.vstack(cartesian_test)
cartesian_test_df = pd.DataFrame(cartesian_test, columns=['shop_id', 'item_id', 'date_block_num'])
cartesian_test_df.head()
cartesian_test_df.shape
from tqdm import tqdm_notebook

def downcast_dtypes(df):
    """
        Changes column types in the dataframe: 
                
                `float64` type to `float32`
                `int64`   type to `int32`
    """
    float_cols = [c for c in df if df[c].dtype == 'float64']
    int_cols = [c for c in df if df[c].dtype == 'int64']
    df[float_cols] = df[float_cols].astype(np.float16)
    df[int_cols] = df[int_cols].astype(np.int16)
    return df
months = train['date_block_num'].unique()
cartesian = []
for month in months:
    shops_in_month = train.loc[train['date_block_num'] == month, 'shop_id'].unique()
    items_in_month = train.loc[train['date_block_num'] == month, 'item_id'].unique()
    cartesian.append(np.array(list(product(*[shops_in_month, items_in_month, [month]])), dtype='int32'))
cartesian_df = pd.DataFrame(np.vstack(cartesian), columns=['shop_id', 'item_id', 'date_block_num'], dtype=np.int32)
cartesian_df.shape
x = train.groupby(['shop_id', 'item_id', 'date_block_num'])['item_cnt_day'].sum().rename('item_cnt_month').reset_index()
x.head()
x.shape
new_train = pd.merge(cartesian_df, x, on=['shop_id', 'item_id', 'date_block_num'], how='left').fillna(0)
new_train[45600:45610]
new_train['item_cnt_month'] = np.clip(new_train['item_cnt_month'], 0, 20)
del x
del cartesian_df
del cartesian
del cartesian_test
del cartesian_test_df
del feb
del jan
del items_test_list
del items_train_list
del train
new_train.sort_values(['date_block_num', 'shop_id', 'item_id'], inplace=True)
new_train.head()
test.insert(loc=3, column='date_block_num', value=34)
test['item_cnt_month'] = 0
test.head()
new_train = new_train.append(test.drop('ID', axis=1))
new_train = pd.merge(new_train, shops, on=['shop_id'], how='left')
new_train = pd.merge(new_train, items.drop('item_name', axis=1), on=['item_id'], how='left')
new_train = pd.merge(new_train, categories.drop('item_category_name', axis=1), on=['item_category_id'], how='left')
new_train.head()

def generate_lag(train, months, lag_column):
    for month in months:
        train_shift = train[['date_block_num', 'shop_id', 'item_id', lag_column]].copy()
        train_shift.columns = ['date_block_num', 'shop_id', 'item_id', lag_column + '_lag_' + str(month)]
        train_shift['date_block_num'] += month
        train = pd.merge(train, train_shift, on=['date_block_num', 'shop_id', 'item_id'], how='left')
    return train
del items
del categories
del shops
del test
new_train = downcast_dtypes(new_train)
import gc
gc.collect()




new_train.head()


new_train.tail()
new_train['month'] = new_train['date_block_num'] % 12
holiday_dict = {0: 6, 1: 3, 2: 2, 3: 8, 4: 3, 5: 3, 6: 2, 7: 8, 8: 4, 9: 8, 10: 5, 11: 4}
new_train['holidays_in_month'] = new_train['month'].map(holiday_dict)
moex = {12: 659, 13: 640, 14: 1231, 15: 881, 16: 764, 17: 663, 18: 743, 19: 627, 20: 692, 21: 736, 22: 680, 23: 1092, 24: 657, 25: 863, 26: 720, 27: 819, 28: 574, 29: 568, 30: 633, 31: 658, 32: 611, 33: 770, 34: 723}
new_train['moex_value'] = new_train.date_block_num.map(moex)
new_train = downcast_dtypes(new_train)
import xgboost as xgb
new_train = new_train[new_train.date_block_num > 11]
import gc
gc.collect()

def fill_na(df):
    for col in df.columns:
        if ('_lag_' in col) & df[col].isnull().any():
            df[col].fillna(0, inplace=True)
    return df
new_train = fill_na(new_train)

def xgtrain():
    regressor = xgb.XGBRegressor(n_estimators=5000, learning_rate=0.01, max_depth=10, subsample=0.5, colsample_bytree=0.5)