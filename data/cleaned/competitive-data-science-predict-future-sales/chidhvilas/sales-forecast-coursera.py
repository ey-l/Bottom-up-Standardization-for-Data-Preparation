import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
sales_train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv', parse_dates=['date'], dayfirst=True)
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
sales_test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
print(f'sales_train columns are{sales_train.columns}', f'items columns are {items.columns}', f'sales_test columns are {sales_test.columns}', f'shops columns are {shops.columns}', sep='\n')
df = sales_train[['item_id', 'item_price']]
df2 = df.drop_duplicates(subset='item_id', keep='first')
df3 = df2.sort_values('item_id')
df3.reset_index(inplace=True, drop=True)
items_updated = pd.merge(items, df3, on='item_id', how='left')
items_updated['item_price'] = items_updated['item_price'].replace(np.nan, 0)
sales_train_1 = pd.merge(sales_train, items, on='item_id', how='inner')
sales_train_1['month_id'] = pd.DatetimeIndex(sales_train_1['date']).month
sales_train_1['year_id'] = pd.DatetimeIndex(sales_train_1['date']).year
sales_train_1.info()
features = ['month_id', 'shop_id', 'item_id', 'item_category_id']
X = sales_train_1[features]
y = sales_train_1['item_cnt_day']
(train_X, val_X, train_y, val_y) = train_test_split(X, y, random_state=0)
criterion = ['mse', 'friedman_mse', 'mae', 'poisson']

def score_mae(train_X, train_y, val_X, val_y):
    model_train = DecisionTreeRegressor(criterion='mse', random_state=0, max_leaf_nodes=55100, min_samples_split=12)