import seaborn as sns
import matplotlib.pyplot as plt

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
train.head()
plt.figure(figsize=(10, 4))
plt.xlim(-100, 3000)
sns.boxplot(x=train.item_cnt_day)
plt.figure(figsize=(10, 4))
plt.xlim(train.item_price.min(), train.item_price.max() * 1.1)
sns.boxplot(x=train.item_price)
train = train[train.item_price < 100000]
train = train[train.item_cnt_day < 750]
num_month = train['date_block_num'].max()
month_list = [i for i in range(num_month + 1)]
shop = []
for i in range(num_month + 1):
    shop.append(5)
item = []
for i in range(num_month + 1):
    item.append(5037)
months_full = pd.DataFrame({'shop_id': shop, 'item_id': item, 'date_block_num': month_list})
train_cleaned = train.drop(labels=['date', 'item_price'], axis=1)
train_cleaned = train_cleaned.groupby(['item_id', 'shop_id', 'date_block_num']).sum().reset_index()
train_cleaned = train_cleaned.rename(index=str, columns={'item_cnt_day': 'item_cnt_month'})
train_cleaned = train_cleaned[['item_id', 'shop_id', 'date_block_num', 'item_cnt_month']]
train_cleaned.tail()
train_cleaned.describe()
clean = pd.merge(train_cleaned, train, how='right', on=['shop_id', 'item_id', 'date_block_num'])
clean = clean.sort_values(by=['date_block_num'])
clean.fillna(0.0, inplace=True)
clean.tail()
import pandas as pd
from sklearn.model_selection import train_test_split
X_full = clean.copy()
X_test_full = clean.copy()
X_full.dropna(axis=0, subset=['item_cnt_month'], inplace=True)
y = X_full.item_cnt_month
X_full.drop(['item_cnt_month'], axis=1, inplace=True)
(X_train_full, X_valid_full, y_train, y_valid) = train_test_split(X_full, y, train_size=0.66, random_state=42)
categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == 'object']
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()
X_train.describe()
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import lightgbm
from lightgbm import LGBMRegressor
from lightgbm import LGBMClassifier
from sklearn.metrics import mean_absolute_error
numerical_transformer = SimpleImputer(strategy='constant')
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_cols), ('cat', categorical_transformer, categorical_cols)])
model = LGBMRegressor()
params = {'feature_fraction': [0.8, 0.5, 0.1], 'max_depth': [2, 4, 5, 10, 15], 'n_estimators': [50, 100, 200]}
from sklearn.model_selection import train_test_split, GridSearchCV