import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
item_categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
sample_submission = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
num_month = train['date_block_num'].max()
month_list = [i for i in range(num_month + 1)]
shop = []
for i in range(num_month + 1):
    shop.append(5)
item = []
for i in range(num_month + 1):
    item.append(5037)
months_full = pd.DataFrame({'shop_id': shop, 'item_id': item, 'date_block_num': month_list})
months_full
train_clean = train.drop(labels=['date', 'item_price'], axis=1)
train_clean = train_clean.groupby(['item_id', 'shop_id', 'date_block_num']).sum().reset_index()
train_clean = train_clean.rename(index=str, columns={'item_cnt_day': 'item_cnt_month'})
train_clean = train_clean[['item_id', 'shop_id', 'date_block_num', 'item_cnt_month']]
train_clean
clean = pd.merge(train_clean, train, how='right', on=['shop_id', 'item_id', 'date_block_num'])
clean = clean.sort_values(by=['date_block_num'])
clean.fillna(0.0, inplace=True)
clean
import pandas as pd
from sklearn.model_selection import train_test_split
X_full = clean.copy()
X_test_full = clean.copy()
X_full.dropna(axis=0, subset=['item_cnt_month'], inplace=True)
y = X_full.item_cnt_month
X_full.drop(['item_cnt_month'], axis=1, inplace=True)
(X_train_full, X_valid_full, y_train, y_valid) = train_test_split(X_full, y, train_size=0.8, test_size=0.2, random_state=0)
categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == 'object']
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
numerical_transformer = SimpleImputer(strategy='constant')
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_cols), ('cat', categorical_transformer, categorical_cols)])
model = RandomForestRegressor(n_estimators=100, random_state=0)
clf = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])