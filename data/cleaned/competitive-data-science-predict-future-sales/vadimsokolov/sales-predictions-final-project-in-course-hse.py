import numpy as np
import pandas as pd
from itertools import product
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
pd.set_option('display.max_columns', 30)
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
items.head()
items.info()
items.describe()
print('Unique values of product names: {}'.format(items.item_name.unique()))
print()
print('Number of unique values of products: {}'.format(items.item_name.nunique()))

def simple_hist(data, x, bins, title, xlabel, xmin, xmax):
    plt.figure(figsize=(12, 8))
    sns.set()
    sns.distplot(data[x], color='lightcoral')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.xlim(xmin, xmax)

simple_hist(items, 'item_category_id', 10, 'Distribution of item categories in the item dataframe', 'item_categories_id', -2, 85)
item_categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
item_categories.head()
item_categories.info()
item_categories.describe()
print('Unique values of product identifiers: {}'.format(item_categories.item_category_id.unique()))
print()
print('Number of unique values of product identifiers: {}'.format(item_categories.item_category_id.nunique()))
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
shops.head()
shops.info()
shops.describe()
print('Unique meanings of store names: {}'.format(shops.shop_name.unique()))
print()
print('Number of unique store values: {}'.format(shops.shop_name.nunique()))
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
train.head()
train.info()
train.describe()
simple_hist(train, 'shop_id', 10, 'Distribution of stores in the train dataframe', 'shop_id', -5, 65)
simple_hist(train, 'item_id', 30, 'Distribution of items in the train dataframe', 'item_id', -1000, 25000)
simple_hist(train, 'item_price', 1000, 'Distribution of the price of items in the train dataframe', 'item_price', -100, 10000)
plt.figure(figsize=(12, 8))
sns.boxplot(y=train['item_price'])
plt.ylim(0, 10000)
plt.grid()
plt.title('Boxplot for the price of goods in the range from 0 to 10000 rubles')
plt.ylabel('item_price')
train.item_cnt_day.value_counts()
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
test.head()
test.info()
test.describe()
simple_hist(test, 'shop_id', 10, 'Shops distributions in the test dataframe', 'id shops', 0, 70)
simple_hist(test, 'item_id', 30, 'Distribution of items in the test dataframe', 'id items', 0, 25000)
sample_submission = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
sample_submission.head()
df_temp = pd.Series(list(train[['item_id', 'shop_id']].itertuples(index=False, name=None)))
test_iter_temp = pd.Series(list(test[['item_id', 'shop_id']].itertuples(index=False, name=None)))
print(str(round(df_temp.isin(test_iter_temp).sum() / len(df_temp), 2) * 100) + '%')
train = train[train.item_price < 100000]
train = train[train.item_cnt_day <= 900]
index_cols = ['shop_id', 'item_id', 'date_block_num']
grid = []
for block_num in train['date_block_num'].unique():
    cur_shops = train.loc[train['date_block_num'] == block_num, 'shop_id'].unique()
    cur_items = train.loc[train['date_block_num'] == block_num, 'item_id'].unique()
    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])), dtype='int32'))
grid = pd.DataFrame(np.vstack(grid), columns=index_cols, dtype=np.int32)
train_merge = train.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_day': 'sum'})
train_merge.columns = ['item_cnt_month']
train_merge.reset_index(inplace=True)
train_merge = pd.merge(grid, train_merge, on=index_cols, how='left').fillna(0)
train_merge['item_cnt_month'] = train_merge['item_cnt_month'].clip(0, 40)
items_prepare = pd.merge(items, item_categories, on='item_category_id')
train_merge = pd.merge(train_merge, items_prepare, on=['item_id'], how='left')
test_temp = test.copy()
test_temp['date_block_num'] = 34
test_temp.drop('ID', axis=1, inplace=True)
test_temp = test_temp.merge(items, how='left', on='item_id')
test_temp = test_temp.merge(item_categories, how='left', on='item_category_id')
test_temp.drop('item_name', axis=1, inplace=True)
train_merge = pd.concat([train_merge, test_temp], axis=0, ignore_index=True, keys=index_cols)
train_merge.fillna(0, inplace=True)
map_dict = {'Чистые носители (штучные)': 'Чистые носители', 'Чистые носители (шпиль)': 'Чистые носители', 'PC ': 'Аксессуары', 'Служебные': 'Служебные '}
train_merge['item_category'] = train_merge['item_category_name'].apply(lambda x: x.split('-')[0])
train_merge['item_category'] = train_merge['item_category'].apply(lambda x: map_dict[x] if x in map_dict.keys() else x)
train_merge['item_category_common'] = LabelEncoder().fit_transform(train_merge['item_category'])
shops['city'] = shops['shop_name'].apply(lambda x: x.split()[0].lower())
shops.loc[shops.city == '!якутск', 'city'] = 'якутск'
shops['city_code'] = LabelEncoder().fit_transform(shops['city'])
coords = dict()
coords['якутск'] = (62.028098, 129.732555, 4)
coords['адыгея'] = (44.609764, 40.100516, 3)
coords['балашиха'] = (55.80945, 37.95806, 1)
coords['волжский'] = (53.43058, 50.119, 3)
coords['вологда'] = (59.2239, 39.88398, 2)
coords['воронеж'] = (51.67204, 39.1843, 3)
coords['выездная'] = (0, 0, 0)
coords['жуковский'] = (55.59528, 38.12028, 1)
coords['интернет-магазин'] = (0, 0, 0)
coords['казань'] = (55.78874, 49.12214, 4)
coords['калуга'] = (54.5293, 36.27542, 4)
coords['коломна'] = (55.07944, 38.77833, 4)
coords['красноярск'] = (56.01839, 92.86717, 4)
coords['курск'] = (51.73733, 36.18735, 3)
coords['москва'] = (55.75222, 37.61556, 1)
coords['мытищи'] = (55.91163, 37.73076, 1)
coords['н.новгород'] = (56.32867, 44.00205, 4)
coords['новосибирск'] = (55.0415, 82.9346, 4)
coords['омск'] = (54.99244, 73.36859, 4)
coords['ростовнадону'] = (47.23135, 39.72328, 3)
coords['спб'] = (59.93863, 30.31413, 2)
coords['самара'] = (53.20007, 50.15, 4)
coords['сергиев'] = (56.3, 38.13333, 4)
coords['сургут'] = (61.25, 73.41667, 4)
coords['томск'] = (56.49771, 84.97437, 4)
coords['тюмень'] = (57.15222, 65.52722, 4)
coords['уфа'] = (54.74306, 55.96779, 4)
coords['химки'] = (55.89704, 37.42969, 1)
coords['цифровой'] = (0, 0, 0)
coords['чехов'] = (55.1477, 37.47728, 4)
coords['ярославль'] = (57.62987, 39.87368, 2)
shops['city_coord_1'] = shops['city'].apply(lambda x: coords[x][0])
shops['city_coord_2'] = shops['city'].apply(lambda x: coords[x][1])
shops['country_part'] = shops['city'].apply(lambda x: coords[x][2])
shops = shops[['shop_id', 'city_code', 'city_coord_1', 'city_coord_2', 'country_part']]
train_merge = pd.merge(train_merge, shops, on=['shop_id'], how='left')
train_merge.drop(['item_name', 'item_category_name', 'item_category'], axis=1, inplace=True)
train_merge.head()

def lag_feature(data, lags, column):
    temp = data[['date_block_num', 'shop_id', 'item_id', column]]
    for lag in lags:
        shifted = temp.copy()
        shifted.columns = ['date_block_num', 'shop_id', 'item_id', column + '_lag_' + str(lag)]
        shifted['date_block_num'] += lag
        data = pd.merge(data, shifted, on=['date_block_num', 'shop_id', 'item_id'], how='left')
        data[column + '_lag_' + str(lag)] = data[column + '_lag_' + str(lag)].astype('float16')
    return data
train_merge = lag_feature(train_merge, [1, 2, 3], 'item_cnt_month')
train_merge.info()

def value_reduction(data):
    for column in data.columns:
        if data[column].dtype == 'float64':
            data[column] = data[column].astype(np.float32)
        if (data[column].dtype == 'int64' or data[column].dtype == 'int32') and (data[column].max() < 32767 and data[column].min() > -32768) and (data[column].isnull().sum() == 0):
            data[column] = data[column].astype(np.int16)
    return data
train_merge = value_reduction(train_merge)
item_id_target_mean = train_merge.groupby(['date_block_num', 'item_id'])['item_cnt_month'].mean().reset_index().rename(columns={'item_cnt_month': 'item_target_enc'}, errors='raise')
train_merge = pd.merge(train_merge, item_id_target_mean, on=['date_block_num', 'item_id'], how='left')
train_merge['item_target_enc'] = train_merge['item_target_enc'].fillna(0).astype(np.float16)
train_merge = lag_feature(train_merge, [1, 2, 3], 'item_target_enc')
train_merge.drop(['item_target_enc'], axis=1, inplace=True)
item_id_target_mean = train_merge.groupby(['date_block_num', 'item_id', 'city_code'])['item_cnt_month'].mean().reset_index().rename(columns={'item_cnt_month': 'item_loc_target_enc'}, errors='raise')
train_merge = pd.merge(train_merge, item_id_target_mean, on=['date_block_num', 'item_id', 'city_code'], how='left')
train_merge['item_loc_target_enc'] = train_merge['item_loc_target_enc'].fillna(0).astype(np.float16)
train_merge = lag_feature(train_merge, [1, 2, 3], 'item_loc_target_enc')
train_merge.drop(['item_loc_target_enc'], axis=1, inplace=True)
item_id_target_mean = train_merge.groupby(['date_block_num', 'item_id', 'shop_id'])['item_cnt_month'].mean().reset_index().rename(columns={'item_cnt_month': 'item_shop_target_enc'}, errors='raise')
train_merge = pd.merge(train_merge, item_id_target_mean, on=['date_block_num', 'item_id', 'shop_id'], how='left')
train_merge['item_shop_target_enc'] = train_merge['item_shop_target_enc'].fillna(0).astype(np.float16)
train_merge = lag_feature(train_merge, [1, 2, 3], 'item_shop_target_enc')
train_merge.drop(['item_shop_target_enc'], axis=1, inplace=True)
first_item_block = train_merge.groupby(['item_id'])['date_block_num'].min().reset_index()
first_item_block['item_first_interaction'] = 1
first_shop_item_buy_block = train_merge[train_merge['date_block_num'] > 0].groupby(['shop_id', 'item_id'])['date_block_num'].min().reset_index()
first_shop_item_buy_block['first_date_block_num'] = first_shop_item_buy_block['date_block_num']
train_merge = pd.merge(train_merge, first_item_block[['item_id', 'date_block_num', 'item_first_interaction']], on=['item_id', 'date_block_num'], how='left')
train_merge = pd.merge(train_merge, first_shop_item_buy_block[['item_id', 'shop_id', 'first_date_block_num']], on=['item_id', 'shop_id'], how='left')
train_merge['first_date_block_num'].fillna(100, inplace=True)
train_merge['shop_item_sold_before'] = (train_merge['first_date_block_num'] < train_merge['date_block_num']).astype('int8')
train_merge.drop(['first_date_block_num'], axis=1, inplace=True)
train_merge['item_first_interaction'].fillna(0, inplace=True)
train_merge['shop_item_sold_before'].fillna(0, inplace=True)
train_merge['item_first_interaction'] = train_merge['item_first_interaction'].astype('int8')
train_merge['shop_item_sold_before'] = train_merge['shop_item_sold_before'].astype('int8')
item_id_target_mean = train_merge[train_merge['item_first_interaction'] == 1].groupby(['date_block_num', 'item_category_id'])['item_cnt_month'].mean().reset_index().rename(columns={'item_cnt_month': 'new_item_cat_avg'}, errors='raise')
train_merge = pd.merge(train_merge, item_id_target_mean, on=['date_block_num', 'item_category_id'], how='left')
train_merge['new_item_cat_avg'] = train_merge['new_item_cat_avg'].fillna(0).astype(np.float16)
train_merge = lag_feature(train_merge, [1, 2, 3], 'new_item_cat_avg')
train_merge.drop(['new_item_cat_avg'], axis=1, inplace=True)
train_merge.isna().sum()
train_merge.fillna(0, inplace=True)
train_merge = train_merge[train_merge['date_block_num'] > 2]
train_merge.to_pickle('train_merge.pkl')
X_train = train_merge[train_merge.date_block_num < 33].drop(['item_cnt_month'], axis=1)
y_train = train_merge[train_merge.date_block_num < 33]['item_cnt_month']
X_valid = train_merge[train_merge.date_block_num == 33].drop(['item_cnt_month'], axis=1)
y_valid = train_merge[train_merge.date_block_num == 33]['item_cnt_month']
X_test = train_merge[train_merge.date_block_num == 34].drop(['item_cnt_month'], axis=1)
print('Shape X_train: {}'.format(X_train.shape))
print()
print('Shape y_train: {}'.format(y_train.shape))
print()
print('Shape X_valid: {}'.format(X_valid.shape))
print()
print('Shape y_valid: {}'.format(y_valid.shape))
print()
print('Shape X_test: {}'.format(X_test.shape))
cat_features = ['country_part', 'item_category_common', 'item_category_id', 'city_code']
catboost = CatBoostRegressor(random_state=1, iterations=2000, verbose=200, depth=4, learning_rate=0.01, l2_leaf_reg=7, max_leaves=2047, min_data_in_leaf=1, subsample=0.7, loss_function='RMSE', eval_metric='RMSE', task_type='GPU', early_stopping_rounds=30, grow_policy='Lossguide', bootstrap_type='Poisson', cat_features=cat_features)