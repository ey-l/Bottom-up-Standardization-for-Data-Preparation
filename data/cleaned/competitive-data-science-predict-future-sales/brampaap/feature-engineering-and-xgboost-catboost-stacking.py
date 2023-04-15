import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from string import punctuation
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

import gc, warnings
warnings.filterwarnings('ignore')

def plot_features(booster, figsize):
    (fig, ax) = plt.subplots(1, 1, figsize=figsize)
    return plot_importance(booster=booster, ax=ax)

def print_files():
    import os
    for (dirname, _, filenames) in os.walk('/kaggle/input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))

def downcast_dtypes(df, inplace=False):
    """
    input  df: object
    output df: object
    
    reject size of col type
    """
    if not inplace:
        data = df.copy()
    else:
        data = df
    float_cols = [c for c in data if data[c].dtype in ['float32', 'float64']]
    int_cols = [c for c in data if data[c].dtype in ['int64', 'int32']]
    data[float_cols] = data[float_cols].astype(np.float16)
    data[int_cols] = data[int_cols].astype(np.int16)
    return data
print_files()
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
sales_train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv', index_col='ID')
sample_submission = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
item_categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
downcast_dtypes(sales_train, inplace=True)
print(sales_train.info())
print('Num of missing values in sales_train: %d' % sales_train.isnull().sum().sum())
print('Num of duplicated rows in sales_train: %d' % sales_train.duplicated().sum())
fig = plt.figure(figsize=(25, 5))
(ax1, ax2) = fig.subplots(1, 2)
ax1.boxplot(x=sales_train.item_price, vert=False)
ax1.set_xlabel('item_price')
ax2.boxplot(x=sales_train.item_cnt_day, vert=False)
ax1.set_xlabel('item_cnt_day')
print('item_price outliers item_id', *sales_train[sales_train.item_price > 45000].index.values)
print('item_cnt_day outliers item_id', *sales_train[sales_train.item_cnt_day > 999].index.values)
print('Price less then zero index', *sales_train[sales_train.item_price < 0].index.values)
med = sales_train[(sales_train.shop_id == 32) & (sales_train.item_id == 11365) & (sales_train.item_price > 0) & (sales_train.date_block_num == 4)].median()
sales_train.iloc[484683] = med
train = sales_train[sales_train.item_price < 100000][sales_train.item_cnt_day < 1001]
print(shops.iloc[np.r_[10, 11, 23, 24, 39, 40, 0, 57, 1, 58]])
d = {0: 57, 1: 58, 10: 11, 23: 24, 39: 40}
shops['shop_id'] = shops['shop_id'].apply(lambda x: d[x] if x in d.keys() else x)
sales_train['shop_id'] = sales_train['shop_id'].apply(lambda x: d[x] if x in d.keys() else x)
test['shop_id'] = test['shop_id'].apply(lambda x: d[x] if x in d.keys() else x)
shops['shop_name_c'] = shops['shop_name'].apply(lambda string: ''.join([pt for pt in string if pt not in punctuation]))
shops['shop_name_c'] = shops['shop_name_c'].str.lower()
shops['shop_type'] = shops['shop_name_c'].apply(lambda x: 'мтрц' if 'мтрц' in x else 'трц' if 'трц' in x else 'трк' if 'трк' in x else 'тц' if 'тц' in x else 'тк' if 'тк' in x else 'тц')
shops['shop_city'] = shops['shop_name_c'].str.partition(' ')[0]
OHE = OneHotEncoder(handle_unknown='ignore', dtype=np.int8)