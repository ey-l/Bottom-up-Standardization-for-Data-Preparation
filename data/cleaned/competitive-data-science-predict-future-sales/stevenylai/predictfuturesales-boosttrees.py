import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
shop = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
item = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
item_category = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
sample_submission = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
train['pd_date'] = pd.to_datetime(train['date'], format='%d.%m.%Y')
train['year'] = pd.DatetimeIndex(train['pd_date']).year
train['month'] = pd.DatetimeIndex(train['pd_date']).month
index_cols = ['shop_id', 'item_id', 'date_block_num']
train_grouped = train.loc[(train['item_price'] < 100000) & (train['item_cnt_day'] <= 900)].drop(columns=['date', 'item_price', 'pd_date']).groupby(index_cols).agg({'item_cnt_day': 'sum'}).reset_index().rename(columns={'item_cnt_day': 'item_cnt_month'})
test['date_block_num'] = 34
train_grouped = pd.concat([train_grouped, test.drop(columns=['ID'])], axis=0, ignore_index=True, keys=index_cols)
print(train_grouped.shape, '\n', train_grouped.head())

def lag_feature(data, lags, column):
    temp = data[index_cols + [column]]
    for lag in lags:
        shifted = temp.copy()
        shifted.columns = index_cols + [column + '_lag_' + str(lag)]
        shifted['date_block_num'] += lag
        data = pd.merge(data, shifted, on=index_cols, how='left')
        data[column + '_lag_' + str(lag)] = data[column + '_lag_' + str(lag)].astype('float16')
    return data
train_lagged = lag_feature(train_grouped, [1, 2, 3], 'item_cnt_month')
train_lagged.fillna(0, inplace=True)
train_lagged = train_lagged.loc[train_lagged['date_block_num'] > 2]
train_lagged.head()
shop['city'] = shop['shop_name'].apply(lambda x: x.split()[0].lower())
shop.loc[shop.city == '!якутск', 'city'] = 'якутск'
shop['city_code'] = LabelEncoder().fit_transform(shop['city'])
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
shop['city_coord_1'] = shop['city'].apply(lambda x: coords[x][0])
shop['city_coord_2'] = shop['city'].apply(lambda x: coords[x][1])
shop['country_part'] = shop['city'].apply(lambda x: coords[x][2])
shop.head()
cat_map = {'Чистые носители (штучные)': 'Чистые носители', 'Чистые носители (шпиль)': 'Чистые носители', 'PC ': 'Аксессуары', 'Служебные': 'Служебные '}
item_category['item_category'] = item_category['item_category_name'].apply(lambda x: x.split('-')[0])
item_category['item_category'] = item_category['item_category'].apply(lambda x: cat_map[x] if x in cat_map.keys() else x)
item_category['item_category_common'] = LabelEncoder().fit_transform(item_category['item_category'])

def value_reduction(data):
    for column in data.columns:
        if data[column].dtype == 'float64':
            data[column] = data[column].astype(np.float32)
        if (data[column].dtype == 'int64' or data[column].dtype == 'int32') and (data[column].max() < 32767 and data[column].min() > -32768) and (data[column].isnull().sum() == 0):
            data[column] = data[column].astype(np.int16)
    return data
all_data = train_lagged.join(item.set_index('item_id'), on='item_id').drop(columns=['item_name']).join(item_category.drop(columns=['item_category_name', 'item_category']).set_index('item_category_id'), on='item_category_id').join(shop.drop(columns=['shop_name', 'city']).set_index('shop_id'), on='shop_id')

def mean_encoding(data, groupby_list, col_list):
    res = data
    for i in range(0, len(groupby_list)):
        groupby = groupby_list[i]
        col = col_list[i]
        index = ['date_block_num'] + groupby
        target_mean = data.groupby(index).agg({'item_cnt_month': 'mean'}).reset_index().rename(columns={'item_cnt_month': col}, errors='raise')
        res = res.join(target_mean.set_index(index), on=index)
        res[col] = res[col].fillna(0).astype(np.float16)
        res = lag_feature(res, [1, 2, 3], col)
        res.drop(columns=[col], axis=1, inplace=True)
    return res
mean_encoded = mean_encoding(all_data, [['item_id'], ['item_id', 'city_code'], ['item_id', 'shop_id']], ['item_target_enc', 'item_loc_target_enc', 'item_shop_target_enc'])
mean_encoded.head()
first_item_block = mean_encoded.groupby(['item_id']).agg({'date_block_num': 'min'}).reset_index()
first_item_block['item_first_sold'] = 1
first_shop_item_buy_block = mean_encoded[mean_encoded['date_block_num'] > 0].groupby(['shop_id', 'item_id']).agg({'date_block_num': 'min'}).reset_index().rename(columns={'date_block_num': 'first_date_block_num'}, errors='raise')
with_interaction = mean_encoded.join(first_item_block.set_index(['item_id', 'date_block_num']), on=['item_id', 'date_block_num']).join(first_shop_item_buy_block.set_index(['item_id', 'shop_id']), on=['item_id', 'shop_id'])
with_interaction['first_date_block_num'].fillna(100, inplace=True)
with_interaction['shop_item_sold_before'] = (with_interaction['first_date_block_num'] < with_interaction['date_block_num']).astype('int8')
with_interaction.drop(['first_date_block_num'], axis=1, inplace=True)
with_interaction['item_first_sold'].fillna(0, inplace=True)
with_interaction['shop_item_sold_before'].fillna(0, inplace=True)
with_interaction['item_first_sold'] = with_interaction['item_first_sold'].astype('int8')
with_interaction['shop_item_sold_before'] = with_interaction['shop_item_sold_before'].astype('int8')
item_id_target_mean = with_interaction[with_interaction['item_first_sold'] == 1].groupby(['date_block_num', 'item_category_id']).agg({'item_cnt_month': 'mean'}).reset_index().rename(columns={'item_cnt_month': 'new_item_cat_avg'}, errors='raise')
with_interaction = with_interaction.join(item_id_target_mean.set_index(['date_block_num', 'item_category_id']), on=['date_block_num', 'item_category_id'])
with_interaction['new_item_cat_avg'] = with_interaction['new_item_cat_avg'].fillna(0).astype(np.float16)
with_interaction = lag_feature(with_interaction, [1, 2, 3], 'new_item_cat_avg')
with_interaction.drop(['new_item_cat_avg'], axis=1, inplace=True)
print(first_item_block.head(), '\n', first_shop_item_buy_block.head(), '\n', item_id_target_mean.head(), '\n', with_interaction.head())
with_interaction.isna().sum()
prepared_data = with_interaction.fillna(0)
X_train = prepared_data.loc[prepared_data['date_block_num'] < 33].drop(columns=['item_cnt_month'])
y_train = prepared_data.loc[prepared_data['date_block_num'] < 33].filter(items=['item_cnt_month'])
X_valid = prepared_data.loc[prepared_data['date_block_num'] == 33].drop(columns=['item_cnt_month'])
y_valid = prepared_data.loc[prepared_data['date_block_num'] == 33].filter(items=['item_cnt_month'])
X_test = prepared_data.loc[prepared_data['date_block_num'] == 34].drop(columns=['item_cnt_month'])
cat_features = ['item_category_id', 'item_category_common', 'city_code', 'country_part']
print(X_train.head(), '\n', X_test.head())
model_cat = CatBoostRegressor(random_state=1, verbose=50, depth=4, learning_rate=0.01, l2_leaf_reg=7, max_leaves=2047, min_data_in_leaf=1, subsample=0.7, loss_function='RMSE', eval_metric='RMSE', early_stopping_rounds=30, grow_policy='Lossguide', cat_features=cat_features)