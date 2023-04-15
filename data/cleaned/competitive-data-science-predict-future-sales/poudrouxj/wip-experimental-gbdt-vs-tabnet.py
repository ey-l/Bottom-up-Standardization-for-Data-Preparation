
import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
submission = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
item_categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
test_dates = train.groupby(['shop_id'])['date_block_num'].max().reset_index()
test_dates['date_block_num'] += 1
test = test.merge(test_dates, how='left', on='shop_id')
train.loc[train.shop_id == 0, 'shop_id'] = 57
test.loc[test.shop_id == 0, 'shop_id'] = 57
train.loc[train.shop_id == 1, 'shop_id'] = 58
test.loc[test.shop_id == 1, 'shop_id'] = 58
train.loc[train.shop_id == 10, 'shop_id'] = 11
test.loc[test.shop_id == 10, 'shop_id'] = 11
test_shops = test.shop_id.unique().tolist()
missing_items = list(np.setdiff1d(test.item_id.unique().tolist(), train.item_id.unique().tolist()))
missing_rows = test[test.item_id.isin(missing_items)].shape[0]
print('Number of items not available in training: ', len(missing_items))
print('Number of rows not available in training: ', missing_rows)
print('Number of rows for testing: ', test.shape[0])
print('Percentage missing: ', missing_rows * 100 / test.shape[0])
train_shop_item = train[['shop_id', 'item_id']].drop_duplicates()
test_shop_item = test[['shop_id', 'item_id']].drop_duplicates()
shop_item = pd.concat([train_shop_item, test_shop_item]).drop_duplicates()
train_shop_dates = train[['shop_id', 'date_block_num']].drop_duplicates()
full = train_shop_dates.merge(shop_item, how='left', on='shop_id')
merged = full.merge(train, how='left', on=['shop_id', 'item_id', 'date_block_num'])
train = merged.loc[:, ['shop_id', 'item_id', 'date', 'date_block_num', 'item_price', 'item_cnt_day']]
train.fillna(0, inplace=True)
items_full = items.merge(item_categories, how='left', on=['item_category_id'])
missing = test[test.item_id.isin(missing_items)].merge(items_full, how='left', on='item_id')
item_cat_map = items[items.item_id.isin(missing_items)][['item_id', 'item_category_id']]
item_cat_map_dict = dict(zip(item_cat_map.item_id.values, item_cat_map.item_category_id.values))
train_monthly = train.groupby(['shop_id', 'item_id', 'date_block_num']).agg(item_sum_mth=pd.NamedAgg(column='item_cnt_day', aggfunc='sum')).reset_index().merge(items_full[['item_id', 'item_category_id']], how='left', on='item_id')
most_pop_item_per_shop_and_cat = train_monthly.groupby(['shop_id', 'item_category_id'])['item_id'].apply(lambda x: x.value_counts().head(1)).reset_index(name='occurences').rename(columns={'level_2': 'item_id'})
most_pop_item_per_shop_and_cat['item_id'] = most_pop_item_per_shop_and_cat['item_id'].astype('int64')
replacement = missing.merge(most_pop_item_per_shop_and_cat, how='left', on=['shop_id', 'item_category_id'], suffixes=('_original', '_replacement'))
print(replacement[replacement.item_id_replacement.isna()].shape, replacement.shape)
print('\n')
train_shop_cat_map = train_monthly.groupby(['shop_id'])['item_category_id'].apply(lambda x: x.unique().astype('int16').tolist()).to_dict()
test_shop_cat_map = test.merge(items[['item_id', 'item_category_id']], how='left', on='item_id').groupby(['shop_id'])['item_category_id'].apply(lambda x: x.unique().astype('int16').tolist()).to_dict()
result = {}
for (shop, item_cat) in test_shop_cat_map.items():
    result[shop] = list(np.setdiff1d(item_cat, train_shop_cat_map[shop]))
replacement_no_nan = missing.merge(most_pop_item_per_shop_and_cat, how='left', on=['item_category_id'], suffixes=('_original', '_replacement'))
replacement.update(replacement_no_nan, overwrite=False)
replacement['item_id_replacement'] = replacement['item_id_replacement'].astype('int16')
replacement['occurences'] = replacement['occurences'].astype('int16')
cols = ['ID', 'shop_id', 'item_id_original', 'item_id_replacement']
test = test.merge(replacement[cols], how='left', left_on=['ID', 'shop_id', 'item_id'], right_on=['ID', 'shop_id', 'item_id_original'], suffixes=('', '_repl'))
test['item_id_new'] = np.where(test.item_id_replacement.isna(), test.item_id, test.item_id_replacement)
test['item_id_new'] = test['item_id_new'].astype('int16')
test.rename(columns={'item_id': 'item_id_org'}, inplace=True)
test.rename(columns={'item_id_new': 'item_id'}, inplace=True)
test.drop(columns=['item_id_original', 'item_id_org', 'item_id_replacement'], inplace=True)
train = train[(train.item_price < 100000) & (train.item_price > 0)]
agg_train = train.groupby(['shop_id', 'item_id', 'date_block_num']).agg(max_price=pd.NamedAgg(column='item_price', aggfunc='max'), min_price=pd.NamedAgg(column='item_price', aggfunc='min'), mean_price=pd.NamedAgg(column='item_price', aggfunc='mean'), item_sum_mth=pd.NamedAgg(column='item_cnt_day', aggfunc='sum'), item_mean_mth=pd.NamedAgg(column='item_cnt_day', aggfunc='mean'), item_std_mth=pd.NamedAgg(column='item_cnt_day', aggfunc=np.nanstd), num_days_sales=pd.NamedAgg(column='date', aggfunc='nunique'))
agg_train = agg_train.reset_index()
agg_train['CATEGORY'] = 'TRAIN'
agg_train['ID'] = -1
test['max_price'] = 0
test['min_price'] = 0
test['mean_price'] = 0
test['item_sum_mth'] = 0
test['item_mean_mth'] = 0
test['item_std_mth'] = 0
test['CATEGORY'] = 'TEST'
test['num_days_sales'] = 0
agg_combined = pd.concat([agg_train, test])
agg_combined['shop_id'] = agg_combined['shop_id'].astype('int32')
agg_combined['item_id'] = agg_combined['item_id'].astype('int32')
agg_combined['date_block_num'] = agg_combined['date_block_num'].astype('int32')
agg_combined['max_price'] = agg_combined['max_price'].astype('float32')
agg_combined['min_price'] = agg_combined['min_price'].astype('float32')
agg_combined['mean_price'] = agg_combined['mean_price'].astype('float32')
agg_combined['item_sum_mth'] = agg_combined['item_sum_mth'].astype('int32')
agg_combined['item_mean_mth'] = agg_combined['item_mean_mth'].astype('float32')
agg_combined['item_std_mth'] = agg_combined['item_std_mth'].astype('float32')
agg_combined['date_block_num_prev'] = agg_combined.sort_values(['shop_id', 'item_id', 'date_block_num']).groupby(['shop_id', 'item_id'])['date_block_num'].shift(-1).values
agg_combined['date_block_num_prev'].fillna(0, inplace=True)
agg_combined['mth_since_l_sale'] = agg_combined['date_block_num'] - agg_combined['date_block_num_prev']

def add_moving_average(df, target, group, windowsize=4, period='D'):
    target_prefix = target.split('_')[0]
    return df.assign(**{'{}_{}_demand_ma_tw{}{}'.format(group[0], target_prefix, windowsize, period): df.groupby(group)[target].rolling(windowsize, min_periods=4).mean().values})

def add_exp_moving_average(df, target, group, windowsize=4, period='D'):
    """Adds exponentially weightings to moving a average over the defined group."""
    target_prefix = target.split('_')[0]
    return df.assign(**{'{}_{}_demand_ema_tw{}{}'.format(group[0], target_prefix, windowsize, period): df.groupby(group)[target].ewm(windowsize, adjust=False, min_periods=4).mean().values})

def add_weighted_moving_average(df, target, group, windowsize=4, period='D'):

    @nb.jit(nopython=True)
    def wma(x):
        x = x.astype(np.float32)
        y = np.arange(1, len(x) + 1).astype(np.float32)
        return np.dot(x, y) / y.sum()
    target_prefix = target.split('_')[0]
    return df.assign(**{'{}_{}_demand_wma_tw{}{}'.format(group[0], target_prefix, windowsize, period): df.groupby(group)[target].rolling(windowsize, min_periods=4).apply(wma, engine='numba', raw=True).values})

def add_moving_std(df, target, group, windowsize=4, period='D'):
    target_prefix = target.split('_')[0]
    return df.assign(**{'{}_{}_demand_mstd_tw{}{}'.format(group[0], target_prefix, windowsize, period): df.groupby(group)[target].rolling(windowsize, min_periods=4).std().values})

def add_group_aggregation(df, target, group, agg):
    """ Add aggregation to dataframe"""
    target_prefix = target.split('_')[0]
    return df.assign(**{'{}_{}_{}_agg_{}'.format(group[0], group[1], target_prefix, agg): df.groupby(group)[target].transform(agg)})
agg_combined.sort_values(['shop_id', 'item_id', 'date_block_num'], inplace=True)
agg_combined['item_cum_sum'] = agg_combined.groupby(['shop_id', 'item_id']).cumsum()['item_sum_mth']
agg_combined['total_item_trend'] = agg_combined.groupby(['item_id', 'date_block_num'])['item_sum_mth'].transform(sum)
agg_combined['totel_item_price_trend'] = agg_combined.groupby(['item_id', 'date_block_num'])['mean_price'].transform(np.mean)
agg_combined['totel_item_price_trend_std'] = agg_combined.groupby(['item_id', 'date_block_num'])['mean_price'].transform(np.std)
agg_combined['date_block_num'] += 1
agg_combined['first_month_sale'] = agg_combined.groupby(['shop_id', 'item_id'])['date_block_num'].transform(min)
agg_combined['month_since_first_sale'] = agg_combined.date_block_num - agg_combined.first_month_sale
agg_combined['total_item_trend_N'] = agg_combined['total_item_trend'] / agg_combined['date_block_num']
agg_combined['item_sum_mth_N'] = agg_combined['item_sum_mth'] / agg_combined['date_block_num']
agg_combined['item_cum_sum_N'] = agg_combined['item_cum_sum'] / agg_combined['date_block_num']
agg_combined['cum_sum_price_N'] = agg_combined.groupby(['shop_id', 'item_id']).cumsum()['mean_price'] / agg_combined['date_block_num']
agg_combined['item_mean_price_N'] = (agg_combined['mean_price'] - agg_combined['totel_item_price_trend']) / agg_combined['totel_item_price_trend']
agg_combined['item_mean_price_N'].fillna(0, inplace=True)
agg_combined['item_revenue_mth'] = agg_combined['mean_price'] * agg_combined['item_sum_mth']
agg_combined_lagged = agg_combined

def create_lag(df, column, lags):
    for lag in lags:
        df[column + '_lagged_' + str(lag)] = df.sort_values(['shop_id', 'item_id', 'date_block_num']).groupby(['shop_id', 'item_id'])[column].shift(-lag)
        df[column + '_lagged_' + str(lag)].fillna(0, inplace=True)
    return df
agg_combined_lagged = create_lag(agg_combined_lagged, 'item_sum_mth', [1, 2, 3, 12])
agg_combined_lagged = create_lag(agg_combined_lagged, 'item_cum_sum', [1, 2, 3, 12])
agg_combined_lagged = create_lag(agg_combined_lagged, 'item_std_mth', [1, 12])
agg_combined_lagged = create_lag(agg_combined_lagged, 'item_mean_mth', [1, 12])
agg_combined_lagged = create_lag(agg_combined_lagged, 'item_std_mth', [1, 12])
agg_combined_lagged = create_lag(agg_combined_lagged, 'num_days_sales', [1, 12])
agg_combined_lagged = create_lag(agg_combined_lagged, 'mean_price', [1])
agg_combined_lagged = create_lag(agg_combined_lagged, 'item_revenue_mth', [1])
agg_combined_lagged = create_lag(agg_combined_lagged, 'item_cum_sum', [1])
agg_combined_lagged = create_lag(agg_combined_lagged, 'cum_sum_price_N', [1])
agg_combined_lagged = create_lag(agg_combined_lagged, 'item_sum_mth_N', [1])
agg_combined_lagged = create_lag(agg_combined_lagged, 'cum_sum_price_N', [1])
agg_combined_lagged = create_lag(agg_combined_lagged, 'total_item_trend_N', [1])
agg_combined_lagged = create_lag(agg_combined_lagged, 'month_since_first_sale', [1])
agg_combined_lagged = create_lag(agg_combined_lagged, 'mth_since_l_sale', [1])
agg_combined_lagged = create_lag(agg_combined_lagged, 'item_mean_price_N', [1])
agg_combined_lagged['sum_cum_sum_ratio_lagged_1'] = agg_combined_lagged['item_sum_mth_N_lagged_1'] / agg_combined_lagged['cum_sum_price_N_lagged_1']
agg_combined_lagged['sum_cum_sum_ratio_lagged_1'].fillna(0, inplace=True)
agg_combined_lagged = agg_combined_lagged.reset_index()
agg_combined_lagged.sort_values(['shop_id', 'item_id', 'date_block_num'])
data = agg_combined_lagged.merge(items[['item_id', 'item_category_id']], how='left', on='item_id')
data['month'] = data['date_block_num'] % 12 + 1
data['year'] = round(data['date_block_num'] / 12)
all_columns = data.columns.tolist()
lagged_columns = [x for x in all_columns if 'lagged' in x]
categorical_columns = ['shop_id', 'item_id', 'item_category_id', 'month', 'year']
numeric_columns = ['date_block_num']
target = ['item_sum_mth']
features = categorical_columns + lagged_columns + numeric_columns
data = data[features + target + ['CATEGORY', 'ID']]
data[target] = data[target].clip(0, 20)
sub_cols = categorical_columns + lagged_columns + numeric_columns + ['ID']
submission = data.loc[data.CATEGORY == 'TEST', sub_cols]
data_cols = categorical_columns + lagged_columns + numeric_columns + target
train_new = data.loc[data.CATEGORY == 'TRAIN', data_cols]
from sklearn.metrics import mean_squared_error

def train_test_val_split_date(df, features, target, val_block, test_block):
    train_x = df.loc[df.date_block_num < val_block, features]
    train_y = df.loc[df.date_block_num < val_block, target]
    val_x = df.loc[(df.date_block_num >= val_block) & (df.date_block_num < test_block), features]
    val_y = df.loc[(df.date_block_num >= val_block) & (df.date_block_num < test_block), target]
    test_x = df.loc[df.date_block_num >= test_block, features]
    test_y = df.loc[df.date_block_num >= test_block, target]
    return (train_x, test_x, val_x, train_y, test_y, val_y)
(train_x, test_x, val_x, train_y, test_y, val_y) = train_test_val_split_date(train_new, features, target, 32, 33)
(train_x.shape, test_x.shape, val_x.shape)
train_x[categorical_columns] = train_x[categorical_columns].astype('str')
test_x[categorical_columns] = test_x[categorical_columns].astype('str')
val_x[categorical_columns] = val_x[categorical_columns].astype('str')
from catboost import Pool, CatBoostRegressor
catboost_params = {'objective': 'Poisson', 'iterations': 50, 'depth': 6, 'learning_rate': 0.2, 'bagging_temperature': 0.2, 'l2_leaf_reg': 9, 'task_type': 'CPU', 'has_time': True}
train_pool = Pool(train_x, train_y, cat_features=categorical_columns)
val_pool = Pool(val_x, val_y, cat_features=categorical_columns)
test_pool = Pool(test_x, test_y, cat_features=categorical_columns)
cat_model = CatBoostRegressor(**catboost_params)