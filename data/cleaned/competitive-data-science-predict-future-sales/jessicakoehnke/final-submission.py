import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import AgglomerativeClustering
import Levenshtein
import xgboost as xgb
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

def import_data():
    """Import all data from csv files"""
    sales = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
    item_cat = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
    items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
    shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
    test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
    return (sales, item_cat, items, shops, test)

def downcast_dtypes(df):
    """Downcast float columns to float32 and int columns to int16"""
    float_cols = [c for c in df if df[c].dtype == 'float64']
    int_cols = [c for c in df if df[c].dtype in ['int64', 'int32']]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    return df
(SALES, ITEM_CAT, ITEMS, SHOPS, TEST) = import_data()

def pre_process_item_data(df):
    """create new item id 'item_lab' to account for duplicates. Of 22k items there are roughly
    100 duplicates"""
    df['name_pre'] = df.item_name.str.replace('[\\*\\!\\./,]', '')
    df['name_pre'] = df.name_pre.str.lower()
    df['name_pre'] = df.name_pre.str.replace('d$', '').str.strip()
    df['item_lab'] = df.name_pre.factorize(sort=True)[0]
pre_process_item_data(ITEMS)
SALES = SALES.merge(ITEMS[['item_id', 'item_category_id', 'item_lab']], how='left', on='item_id')
SALES.drop(['item_id', 'date', 'item_category_id'], axis=1, inplace=True)

def clip_fillna_prices(df):
    quantile_99 = df.item_price.quantile(0.99)
    df.item_price.clip(upper=quantile_99, inplace=True)
    idx = df[df.item_price < 0].index.tolist()
    df.at[idx[0], 'item_price'] = (2499.0 + 1249.0) / 2.0

def aggregate_item_count(df):
    train = df.groupby(['date_block_num', 'shop_id', 'item_lab']).agg(item_price=('item_price', 'mean'), item_cnt_month=('item_cnt_day', 'sum')).reset_index()
    train.item_cnt_month.clip(lower=0, upper=20, inplace=True)
    return train

def add_missing_item_shop_pairs(df, price):
    df.set_index(['date_block_num', 'shop_id', 'item_lab'], inplace=True)
    idx = []
    for month in df.index.unique('date_block_num'):
        shops_unique = df.loc[month].index.unique('shop_id')
        items_unique = df.loc[month].index.unique('item_lab')
        idx.append(pd.MultiIndex.from_product([[month], shops_unique, items_unique], names=['date_block_num', 'shop_id', 'item_lab']))
    idx = idx[0].append(idx[1:])
    df = df.reindex(idx, fill_value=0.0)
    df = df.reset_index()
    df = pd.merge(df, price, how='left', on='item_lab', suffixes=('', '_'))
    df['item_price'] = np.where(df.item_price > 0, df.item_price, df.item_price_)
    del df['item_price_']
    return df
clip_fillna_prices(SALES)
PRICE = SALES[['item_lab', 'item_price']][SALES.item_price > 0].groupby('item_lab').agg('mean')
MATRIX = aggregate_item_count(SALES)
MATRIX = add_missing_item_shop_pairs(MATRIX, PRICE)
MATRIX = MATRIX.merge(ITEMS[['item_category_id', 'item_lab']].drop_duplicates(), how='left', on='item_lab')
TEST['date_block_num'] = 34
TEST.drop('ID', axis=1, inplace=True)
TEST['item_cnt_month'] = np.nan
TEST = TEST.merge(ITEMS[['item_id', 'item_lab', 'item_category_id']], how='left', on='item_id')
TEST = pd.merge(TEST.drop('item_id', axis=1), PRICE, how='left', on='item_lab')
MATRIX = MATRIX.append(TEST, ignore_index=True)
MATRIX['item_category_id'] = MATRIX.item_category_id.astype('int16')

def fillna_means(df, col):
    """Fill with means of item_lab, if not available, use means of category"""
    df[col] = df.groupby('item_lab')[col].transform(lambda x: x.fillna(x.mean()))
    df[col] = df.groupby('item_category_id')[col].transform(lambda x: x.fillna(x.mean()))
fillna_means(MATRIX, 'item_price')

def pre_process_item_cat_data(df):
    """Do pre processing of item_cat data frame"""
    cat = df.item_category_name.str.split('-', n=1, expand=True)
    df['cat1'] = cat[0].str.strip().str.lower()
    df['cat2'] = cat[1].str.strip().str.lower()
    df.cat1.fillna('', inplace=True)
    df.cat2.fillna('', inplace=True)
    df['cat1_lab'] = df.cat1.factorize(sort=True)[0]
    df['cat2_lab'] = df.cat2.factorize(sort=True)[0]

def get_string_correlation(array):

    def dist(x, y):
        lev = Levenshtein.distance(x[0], y[0])
        m = np.mean([len(x[0]), len(y[0])])
        return lev / m
    cor = pdist(array.reshape(-1, 1), dist)
    return squareform(cor)

def agglomerative_clustering(distance_matrix, num_clusters=None, threshold=None):
    model = AgglomerativeClustering(n_clusters=num_clusters, affinity='precomputed', linkage='average', distance_threshold=threshold)