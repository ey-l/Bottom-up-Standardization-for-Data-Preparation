import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
import time
import pandas as pd
item_categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
sales_train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
sample_submission = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
sales_train['date'] = pd.to_datetime(sales_train['date'], format='%d.%m.%Y')
sales_train = sales_train.assign(day=sales_train['date'].dt.day, month=sales_train['date'].dt.month, year=sales_train['date'].dt.year)

def data_cleaner(df):
    X = df[['date_block_num', 'shop_id', 'item_id', 'item_price']]
    y = df['item_cnt_day']
    all_data = X.join(y)
    a = all_data.groupby(['date_block_num', 'shop_id', 'item_id']).sum()
    a['item_price'] = all_data.groupby(['date_block_num', 'shop_id', 'item_id'])['item_price'].mean()
    a.reset_index(inplace=True)
    a.rename({'item_cnt_day': 'item_cnt_month'}, inplace=True, axis=1)
    all_data = a
    X = all_data.drop('item_cnt_month', axis=1)
    y = all_data['item_cnt_month']
    all_data['item_category_id'] = all_data['item_id'].map(items['item_category_id'])
    all_data = all_data[['date_block_num', 'shop_id', 'item_id', 'item_category_id', 'item_price', 'item_cnt_month']]
    return all_data

def model_evaluator(ad):
    model = LinearRegression()
    curr_block_num = 33
    X_train = ad.loc[ad['date_block_num'] < 33].drop(['item_cnt_month'], axis=1)
    y_train = ad.loc[ad['date_block_num'] < 33]['item_cnt_month']
    X_valid = ad.loc[ad['date_block_num'] == 33].drop(['item_cnt_month'], axis=1)
    y_valid = ad.loc[ad['date_block_num'] == 33]['item_cnt_month']