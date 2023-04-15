import lightgbm
import numpy as np
import os

def check_LGBM_gpu_support():
    data = np.random.rand(50, 2)
    label = np.random.randint(2, size=50)
    train_data = lightgbm.Dataset(data, label=label)
    params = {'device': 'gpu'}
    try:
        gbm = lightgbm.train(params, num_boost_round=1, train_set=train_data)
        return True
    except Exception as e:
        return False
if not check_LGBM_gpu_support():


    print('Kernel should be Restarted!')
    os._exit(0)
else:
    print('LGBM GPU version installed!')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import gc
from sklearn.preprocessing import LabelEncoder
from matplotlib.pylab import rcParams
from xgboost import XGBRegressor
sns.set(style='darkgrid')
sns.set(rc={'figure.figsize': (15, 8)})
aggregation_cols = ['date_block_num', 'shop_id', 'item_id']
PATH = '../input/predict-future-sales-project-final/'
Kaggle_PATH = '/kaggle/input/competitive-data-science-predict-future-sales/'
items = pd.read_csv(Kaggle_PATH + 'items.csv')
shops = pd.read_csv(Kaggle_PATH + 'shops.csv')
cats = pd.read_csv(Kaggle_PATH + 'item_categories.csv')
train = pd.read_csv(Kaggle_PATH + 'sales_train.csv')
test = pd.read_csv(Kaggle_PATH + 'test.csv')
USE_SERIALIZED_MODELS = True
ENSEMBLING_METHOD = 2
START_FROM_DICT = {'run_all': 0, 'before_feature_engineering': 1, 'after_train_valid_split': 2, 'after_mean_encoding': 3}
START_FROM = 0
if START_FROM == 0:
    print(train.head())
if START_FROM == 0:
    print(train.describe())
if START_FROM == 0:
    print(train.isna().any())
if START_FROM == 0:
    train['item_price'].plot()
if START_FROM == 0:
    print(train[train['item_price'] > 50000]['item_id'].unique(), train[train['item_price'] > 50000]['item_id'].count(), train[train['item_price'] < 0]['item_id'].count())
if START_FROM == 0:
    item_transaction_count = train.groupby(['item_id'])['date'].count()
    print(item_transaction_count[item_transaction_count < 10].value_counts())
if START_FROM == 0:
    item_max_date_block = train.groupby(['item_id'])['date_block_num'].max()
    item_min_date_block = train.groupby(['item_id'])['date_block_num'].min()
    item_nb_transactions_last_date_block = pd.concat([item_transaction_count, item_max_date_block, item_min_date_block], axis=1)
    item_nb_transactions_last_date_block.columns = ['nbr_transactions', 'last_sale', 'first_sale']
    item_nb_transactions_last_date_block['diff_first_last'] = item_nb_transactions_last_date_block['last_sale'] - item_nb_transactions_last_date_block['first_sale']
    print(item_nb_transactions_last_date_block[(item_nb_transactions_last_date_block['nbr_transactions'] < 10) & (item_nb_transactions_last_date_block['last_sale'] < 22)])
if START_FROM == 0:
    train['item_cnt_day'].plot()
if START_FROM == 0:
    print(len(train[train['item_cnt_day'] < 0]['item_id'].unique()), train[train['item_cnt_day'] < 0]['item_id'].count(), train[train['item_cnt_day'] > 750]['item_id'].count())
if START_FROM == 0:
    print(shops)
if START_FROM == 0:
    print(cats.shape)
    print(cats.head(20))
if START_FROM == 0:
    cats_type_code = cats.item_category_name.apply(lambda x: x.split(' ')[0]).astype(str)
    print(cats_type_code.value_counts())
if START_FROM == 0:
    print(items.head(20))
    print(items.shape)
if START_FROM == 0:
    df = train[['date_block_num', 'shop_id', 'item_id', 'item_cnt_day']]
    pivot_cnt_month_per_shop = pd.pivot_table(data=df, index=['date_block_num'], columns=['shop_id'], fill_value=0, values='item_cnt_day', aggfunc='sum')
    pivot_cnt_month_per_shop.reset_index(drop=True, inplace=True)
    pivot_cnt_month_per_shop.columns.name = None

    print(pivot_cnt_month_per_shop)
if START_FROM == 0:
    plt.figure()
    ax = pivot_cnt_month_per_shop.plot()
    ax.legend(loc=(1.01, 0.01), ncol=2)
    plt.tight_layout()
if START_FROM == 0:
    print(np.sort(test['shop_id'].unique()))
if START_FROM == 0:
    df = pivot_cnt_month_per_shop.cumsum()
    plt.figure()
    ax = df.plot()
    ax.legend(loc=(1.01, 0.01), ncol=2)
    plt.tight_layout()
if START_FROM == 0:
    last_sum = df.iloc[33]
    print(last_sum[last_sum > 100000])
if START_FROM == 0:
    train['revenue'] = train['item_price'] * train['item_cnt_day']
    pivot_revenue_month_per_shop = pd.pivot_table(train, values='revenue', index=['date_block_num'], columns=['shop_id'], aggfunc=np.sum, fill_value=0)
    plt.figure()
    ax = pivot_revenue_month_per_shop.plot()
    ax.legend(loc=(1.01, 0.01), ncol=2)
    plt.tight_layout()
if START_FROM == 0:
    train = train[(train['item_price'] < 50000) & (train['item_cnt_day'] < 1000) & (train['item_price'] > 0)]
    item_transaction_count = train.groupby(['item_id'])['date'].count()
    item_max_date_block = train.groupby(['item_id'])['date_block_num'].max()
    item_nb_transactions_last_date_block = pd.concat([item_transaction_count, item_max_date_block], axis=1)
    item_nb_transactions_last_date_block.columns = ['nbr_transactions', 'last_sale']
    items_to_drop = item_nb_transactions_last_date_block[(item_nb_transactions_last_date_block['nbr_transactions'] < 10) & (item_nb_transactions_last_date_block['last_sale'] < 22)]
    print('Shapes before the drop: ', train.shape, items.shape, items_to_drop.shape)
    train = train[~train['item_id'].isin(list(items_to_drop.index))]
    print('Shapes after the drop: ', train.shape)
if START_FROM == 0:
    train['shop_id'] = train['shop_id'].replace({0: 57, 1: 58, 11: 10, 40: 39})
    train = train[~train['shop_id'].isin([8, 9, 20, 23, 32])]
    test['shop_id'] = test['shop_id'].replace({0: 57, 1: 58, 11: 10, 40: 39})
    train.reset_index(drop=True, inplace=True)
    print('Shapes after the drop: ', train.shape)
if START_FROM == 0:
    shops.loc[shops['shop_name'] == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'
    shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])
    shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'
    shops['shop_city'] = LabelEncoder().fit_transform(shops['city'])
    shops['category'] = shops['shop_name'].str.split(' ').map(lambda x: x[1])
    shop_location_dict = {'ТК': 1, 'ТЦ': 4, 'ТРК': 2, 'ТРЦ': 3, 'МТРЦ': 0}
    shops['shop_category'] = shops['category'].apply(lambda x: shop_location_dict[x] if x in shop_location_dict else 0)
    shops = shops[['shop_id', 'shop_category', 'shop_city']]
if START_FROM == 0:
    cats['sub_type_code1'] = cats.item_category_name.apply(lambda x: x.split(' ')[0]).astype(str)
    category = []
    for cat in cats['sub_type_code1'].unique():
        if len(cats[cats['sub_type_code1'] == cat]) >= 5:
            category.append(cat)
    cats['sub_type_code1'] = cats['sub_type_code1'].apply(lambda x: x if x in category else 'others')
    cats['type_code'] = cats['sub_type_code1']
    cats['sub_type_code2'] = cats['sub_type_code1']
    cats.loc[(cats['sub_type_code1'] == 'Игровые') | (cats['sub_type_code1'] == 'Аксессуары'), 'type_code'] = 'Игры'
    cats.loc[cats['sub_type_code1'] == 'Игры', 'sub_type_code2'] = cats['item_category_name'].str[:8]
    cats.loc[cats['sub_type_code1'] == 'Аксессуары', 'sub_type_code2'] = cats['item_category_name'].str[:15]
    cats.loc[cats['sub_type_code1'] == 'Игровые', 'sub_type_code2'] = cats['item_category_name'].str[:20]
    print(cats)
if START_FROM == 0:
    cats['type_code'] = LabelEncoder().fit_transform(cats['type_code'])
    cats['sub_type_code1'] = LabelEncoder().fit_transform(cats['sub_type_code1'])
    cats['sub_type_code2'] = LabelEncoder().fit_transform(cats['sub_type_code2'])
    cats = cats[['item_category_id', 'sub_type_code1', 'sub_type_code2', 'type_code']]
if START_FROM == 0:
    import spacy

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import PCA

    def preprocess_items(items, nb_components=4):

        class LemmaTokenizer(object):

            def __init__(self):
                self.spacynlp = spacy.load('ru_core_news_md')

            def __call__(self, doc):
                nlpdoc = self.spacynlp(doc)
                nlpdoc = [token.lemma_.strip() for token in nlpdoc if len(token.lemma_.strip()) > 1 or token.lemma_.strip().isalnum()]
                return nlpdoc
        vect = TfidfVectorizer(tokenizer=LemmaTokenizer(), stop_words='english')
        features = vect.fit_transform(items['item_name'])
        pca = PCA(n_components=nb_components, random_state=999)
        reduced_features = pca.fit_transform(features.toarray())
        for i in range(nb_components):
            items['items_pca' + str(i)] = reduced_features[:, i]
        return items.drop('item_name', axis=1)
    items_nb_components = 6
    if items_nb_components != 0:
        items = preprocess_items(items, items_nb_components)
    else:
        items.drop('item_name', axis=1, inplace=True)
if START_FROM == 0:
    from itertools import product
    matrix = []
    for i in range(34):
        sales = train[train.date_block_num == i]
        matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype=np.int16))
    matrix = pd.DataFrame(np.vstack(matrix), columns=aggregation_cols)
    matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)
    matrix['shop_id'] = matrix['shop_id'].astype(np.int8)
    matrix['item_id'] = matrix['item_id'].astype(np.int16)
    matrix.sort_values(aggregation_cols, inplace=True)
    group = train.groupby(aggregation_cols).agg({'item_cnt_day': ['sum']})
    group.columns = ['item_cnt_month']
    group.reset_index(inplace=True)
    matrix = pd.merge(matrix, group, on=aggregation_cols, how='left')
    matrix['item_cnt_month'] = matrix['item_cnt_month'].fillna(0).astype(np.float16)
    test['date_block_num'] = 34
    test['date_block_num'] = test['date_block_num'].astype(np.int8)
    test['shop_id'] = test.shop_id.astype(np.int8)
    test['item_id'] = test.item_id.astype(np.int16)
    matrix = pd.concat([matrix, test.drop(['ID'], axis=1)], ignore_index=True, sort=False, keys=aggregation_cols)
    matrix.fillna(0, inplace=True)
    matrix = pd.merge(matrix, shops, on=['shop_id'], how='left')
    matrix = pd.merge(matrix, items, on=['item_id'], how='left')
    matrix = pd.merge(matrix, cats, on=['item_category_id'], how='left')
    matrix['shop_city'] = matrix['shop_city'].astype(np.int8)
    matrix['shop_category'] = matrix['shop_category'].astype(np.int8)
    matrix['item_category_id'] = matrix['item_category_id'].astype(np.int8)
    matrix['sub_type_code1'] = matrix['sub_type_code1'].astype(np.int8)
    matrix['sub_type_code2'] = matrix['sub_type_code2'].astype(np.int8)
    matrix['type_code'] = matrix['type_code'].astype(np.int8)
    for i in range(items_nb_components):
        matrix['items_pca' + str(i)] = matrix['items_pca' + str(i)].astype(np.float16)
if START_FROM == 0:
    pickle.dump(matrix, open('matrix_00_init_10Feb.pkl', 'wb'))
    print('Before Feature Engineering Executed!!')
else:
    print('Jumped to Feature Engineering START_FROM>0')
if START_FROM == 1:
    matrix = pickle.load(open(PATH + 'matrix_00_init_10Feb.pkl', 'rb'))

def lag_feature(df, lags, cols):
    for col in cols:
        tmp = df[aggregation_cols + [col]]
        for i in lags:
            shifted = tmp.copy()
            shifted.columns = aggregation_cols + [col + '_lag_' + str(i)]
            shifted['date_block_num'] = shifted['date_block_num'] + i
            df = pd.merge(df, shifted, on=aggregation_cols, how='left')
    return df

def add_feature_using_lag(src_df, dest_df, feature_name, groupby_list, aggregate_item, aggregate_func_list, lag_list, with_drop=True, np_type=np.float16):
    group = src_df.groupby(groupby_list).agg({aggregate_item: aggregate_func_list})
    group.columns = [feature_name]
    group.reset_index(inplace=True)
    dest_df = dest_df.merge(group, on=groupby_list, how='left')
    dest_df[feature_name] = dest_df[feature_name].astype(np_type)
    if lag_list != 0:
        dest_df = lag_feature(dest_df, lag_list, [feature_name])
    if with_drop:
        dest_df.drop([feature_name], axis=1, inplace=True)
    return dest_df
if START_FROM < 2:
    print('1- Adding item_cnt_month lag features.')
    matrix = lag_feature(matrix, [1, 2, 3], ['item_cnt_month'])
    print('2- Adding the previous month average item_cnt.')
    matrix = add_feature_using_lag(matrix, matrix, 'date_avg_item_cnt', ['date_block_num'], 'item_cnt_month', ['mean'], [1])
    print('3- Adding lag values of item_cnt_month for month / item_id.')
    matrix = add_feature_using_lag(matrix, matrix, 'date_item_avg_item_cnt', ['date_block_num', 'item_id'], 'item_cnt_month', ['mean'], [1, 2, 3])
    print('4- Adding lag values for item_cnt_month for every month / shop combination.')
    matrix = add_feature_using_lag(matrix, matrix, 'date_shop_avg_item_cnt', ['date_block_num', 'shop_id'], 'item_cnt_month', ['mean'], [1, 2, 3])
    print('5- Adding lag values for item_cnt_month for month/shop/item.')
    matrix = add_feature_using_lag(matrix, matrix, 'date_shop_item_avg_item_cnt', ['date_block_num', 'shop_id', 'item_id'], 'item_cnt_month', ['mean'], [1, 2, 3])
    print('6- Adding lag values for item_cnt_month for month/shop/item subtype1.')
    matrix = add_feature_using_lag(matrix, matrix, 'date_shop_subtype1_avg_item_cnt', ['date_block_num', 'shop_id', 'sub_type_code1'], 'item_cnt_month', ['mean'], [1])
    print('7- Adding lag values for item_cnt_month for month/shop/item subtype2.')
    matrix = add_feature_using_lag(matrix, matrix, 'date_shop_subtype2_avg_item_cnt', ['date_block_num', 'shop_id', 'sub_type_code2'], 'item_cnt_month', ['mean'], [1])
    print('8- Adding lag values for item_cnt_month for month/city.')
    matrix = add_feature_using_lag(matrix, matrix, 'date_city_avg_item_cnt', ['date_block_num', 'shop_city'], 'item_cnt_month', ['mean'], [1])
    print('9- Adding lag values for item_cnt_month for month/city/item.')
    matrix = add_feature_using_lag(matrix, matrix, 'date_item_city_avg_item_cnt', ['date_block_num', 'item_id', 'shop_city'], 'item_cnt_month', ['mean'], [1])
if START_FROM < 2:
    print('1- Adding average item price')
    matrix = add_feature_using_lag(train, matrix, 'item_avg_item_price', ['item_id'], 'item_price', ['mean'], [], with_drop=False)
    print('2- Adding item price average per month')
    matrix = add_feature_using_lag(train, matrix, 'date_item_avg_item_price', ['date_block_num', 'item_id'], 'item_price', ['mean'], [1, 2, 3], with_drop=False)
    print('3- Adding avg item price per month change over previous months (lag values)')
    lags = [1, 2, 3]
    for i in lags:
        matrix['delta_price_lag_' + str(i)] = (matrix['date_item_avg_item_price_lag_' + str(i)] - matrix['item_avg_item_price']) / matrix['item_avg_item_price']

    def select_trends(row):
        for i in lags:
            if row['delta_price_lag_' + str(i)]:
                return row['delta_price_lag_' + str(i)]
        return 0
    matrix['delta_price_lag'] = matrix.apply(select_trends, axis=1)
    matrix['delta_price_lag'] = matrix['delta_price_lag'].astype(np.float16)
    matrix['delta_price_lag'].fillna(0, inplace=True)
    features_to_drop = ['item_avg_item_price', 'date_item_avg_item_price']
    for i in lags:
        features_to_drop.append('date_item_avg_item_price_lag_' + str(i))
        features_to_drop.append('delta_price_lag_' + str(i))
    matrix.drop(features_to_drop, axis=1, inplace=True)
if START_FROM < 2:
    train['revenue'] = train['item_cnt_day'] * train['item_price']
    print('1- Adding total shop revenue per month.')
    matrix = add_feature_using_lag(train, matrix, 'date_shop_revenue', ['date_block_num', 'shop_id'], 'revenue', ['sum'], [], with_drop=False, np_type=np.float32)
    matrix = add_feature_using_lag(matrix, matrix, 'shop_avg_revenue', ['shop_id'], 'date_shop_revenue', ['mean'], [], with_drop=False, np_type=np.float32)
    matrix['delta_revenue'] = (matrix['date_shop_revenue'] - matrix['shop_avg_revenue']) / matrix['shop_avg_revenue']
    matrix['delta_revenue'] = matrix['delta_revenue'].astype(np.float32)
    matrix = lag_feature(matrix, [1], ['delta_revenue'])
    matrix['delta_revenue_lag_1'] = matrix['delta_revenue_lag_1'].astype(np.float32)
    matrix.drop(['date_shop_revenue', 'shop_avg_revenue', 'delta_revenue'], axis=1, inplace=True)
if START_FROM < 2:

    def nb_business_days(y, m):
        current_month = str(y) + '-' + ('0' if m < 10 else '') + str(m)
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1
        next_month = str(y) + '-' + ('0' if m < 10 else '') + str(m)
        return np.busday_count(current_month, next_month)
    matrix['month'] = matrix['date_block_num'] % 12
    matrix['year'] = matrix['date_block_num'] // 12
    days = pd.Series([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    matrix['days'] = matrix['month'].map(days).astype(np.int8)
    matrix['year'] = matrix['year'].astype('int8')
    tmp_series = pd.Series(np.linspace(0, 34, 35), dtype=np.int8)
    nb_business_days_arr = tmp_series.apply(lambda x: nb_business_days(x // 12 + 2012, x % 12 + 1))
    matrix['nb_business_days'] = matrix['date_block_num'].map(nb_business_days_arr).astype(np.int8)
    matrix['nb_holiday_days'] = matrix['days'] - matrix['nb_business_days']
    matrix['nb_holiday_days'] = matrix['nb_holiday_days'].astype(np.int8)
    matrix['item_shop_first_sale'] = matrix['date_block_num'] - matrix.groupby(['item_id', 'shop_id'])['date_block_num'].transform('min')
    matrix['item_first_sale'] = matrix['date_block_num'] - matrix.groupby(['item_id'])['date_block_num'].transform('min')
    matrix = matrix[matrix['date_block_num'] > 3]
if START_FROM < 2:
    data = matrix.copy()
    del matrix
    gc.collect()
    X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
    Y_train = data[data.date_block_num < 33]['item_cnt_month']
    X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
    Y_valid = data[data.date_block_num == 33]['item_cnt_month']
    X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)
    X_train.fillna(0, inplace=True)
    X_valid.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)
    Y_train = Y_train.clip(0, 20).fillna(0).astype(np.float32)
    Y_valid = Y_valid.clip(0, 20).fillna(0).astype(np.float32)
    del data
if START_FROM < 2:
    pickle.dump((X_train, Y_train, X_valid, Y_valid, X_test), open('training_datasets_00_10Feb.pkl', 'wb'))
    print('Until Train_Valid_Split Executed!!')
else:
    print('Jumped After Train_Valid_Split!!')
if START_FROM == 2:
    (X_train, Y_train, X_valid, Y_valid, X_test) = pickle.load(open(PATH + 'training_datasets_00_10Feb.pkl', 'rb'))
    X_train.fillna(0, inplace=True)
    X_valid.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)
    Y_train = Y_train.clip(0, 20).fillna(0).astype(np.float32)
    Y_valid = Y_valid.clip(0, 20).fillna(0).astype(np.float32)
    gc.collect()
if START_FROM < 3:
    print(Y_valid.mean())
cst = 0.285
sample = test[['ID']].copy()
sample['item_cnt_month'] = cst

last_month_cnt = train[train['date_block_num'] == 33].groupby(['shop_id', 'item_id'])['item_cnt_day'].sum().reset_index()
last_month_cnt.rename(columns={'item_cnt_day': 'item_cnt_month'}, inplace=True)
df_test = pd.merge(test, last_month_cnt, how='left', on=['shop_id', 'item_id'])
df_test['item_cnt_month'] = df_test['item_cnt_month'].fillna(0).clip(0, 20)

nb_months = 3
last_month_cnt = train[train['date_block_num'] > 33 - nb_months].groupby(['shop_id', 'item_id'])['item_cnt_day'].sum().reset_index()
last_month_cnt.rename(columns={'item_cnt_day': 'item_cnt_month'}, inplace=True)
last_month_cnt['item_cnt_month'] = last_month_cnt['item_cnt_month'] / nb_months
df_test = pd.merge(test, last_month_cnt, how='left', on=['shop_id', 'item_id'])
df_test['item_cnt_month'] = df_test['item_cnt_month'].fillna(0).clip(0, 20)

gc.collect()
from xgboost import plot_importance

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def plot_features(booster, figsize):
    rcParams['figure.figsize'] = (12, 4)
    (fig, ax) = plt.subplots(1, 1, figsize=figsize)
    return plot_importance(booster=booster, ax=ax)

def print_features(model):
    feat_imp = model.feature_importances_
    imp_df = pd.Series(feat_imp, index=list(X_train.columns))
    print(imp_df.sort_values(ascending=False))
params00 = {'max_depth': 10, 'eta': 0.01}
params01 = {'max_depth': 9, 'eta': 0.01, 'min_child_weight': 10, 'subsample': 0.6, 'colsample_bytree': 0.6, 'alpha': 60, 'reg_lambda': 10, 'booster': 'gbtree', 'gamma': 0.1, 'grow_policy': 'depthwise'}

def run_XGB_training(params, with_submission=True, with_val_preds=False, with_train_preds=False, show_feat=False, use_file=False, fname=''):
    if use_file:
        model = pickle.load(open(PATH + fname, 'rb'))
    else:
        model = XGBRegressor(tree_method='gpu_hist', seed=42, n_estimators=2000, **params)