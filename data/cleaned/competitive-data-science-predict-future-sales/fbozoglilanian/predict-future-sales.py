import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
X_full = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
X_test_full = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
item_categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
X_full = pd.merge(X_full, X_test_full, on=['shop_id', 'item_id'], how='inner', sort=False)
print(X_full.head())
print(items.head())
print(item_categories.head())
items = pd.merge(items, item_categories, on='item_category_id', how='left')
X_test_full = pd.merge(X_test_full, items, how='left', on='item_id')
X_test_full = X_test_full[['ID', 'shop_id', 'item_id', 'item_category_id']]
X_test_full
items = pd.merge(items, item_categories, on='item_category_id', how='left')
X_full = pd.merge(X_full, items, how='left', on='item_id')
X_full
X_full = X_full.astype({'ID': 'int'})
X_set = X_full.groupby(['ID', 'shop_id', 'item_id', 'item_category_id']).mean().reset_index()
X_set = X_set.rename(columns={'item_cnt_day': 'item_cnt_month'}, inplace=False)
X_set.drop(['item_price', 'date_block_num'], axis=1, inplace=True)
X_set
from sklearn.model_selection import train_test_split
X_set.dropna(axis=0, subset=['item_cnt_month'], inplace=True)
y = X_set['item_cnt_month']
X_set.drop(['item_cnt_month'], axis=1, inplace=True)
(X_train_full, X_valid_full, y_train, y_valid) = train_test_split(X_set, y, train_size=0.9, test_size=0.1, random_state=0)
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]
my_cols = numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
numerical_transformer = SimpleImputer(strategy='median')
preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_cols)])
from xgboost import XGBRegressor
model = XGBRegressor(n_estimators=300, learning_rate=0.002, n_jobs=5)
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])