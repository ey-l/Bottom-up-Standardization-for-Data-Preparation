import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
import seaborn as sns
from sklearn import metrics
from scipy import stats
from copy import deepcopy
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import accuracy_score, mean_squared_error
train_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
sub_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
shops_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
items_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
item_categories_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
train_df.head()
test_df.head()
shops_df.head()
items_df.head()
train_df.describe()
train_df.isnull().sum()
train_df.value_counts
train_df.shape
train_df.drop(['date_block_num', 'item_price'], axis=1, inplace=True)
train_df['date'] = pd.to_datetime(train_df['date'], dayfirst=True)
train_df['date'] = train_df['date'].apply(lambda x: x.strftime('%Y-%m'))
train_df.head()
df = train_df.groupby(['date', 'shop_id', 'item_id']).sum()
df = df.pivot_table(index=['shop_id', 'item_id'], columns='date', values='item_cnt_day', fill_value=0)
df.reset_index(inplace=True)
df.head()
test_df = pd.merge(test_df, df, on=['shop_id', 'item_id'], how='left')
test_df.drop(['ID', '2013-01'], axis=1, inplace=True)
test_df = test_df.fillna(0)
test_df.head()
Y_train = df['2015-10'].values
X_train = df.drop(['2015-10'], axis=1)
X_test = test_df
print(X_train.shape, Y_train.shape)
print(X_test.shape)
(x_train, x_test, y_train, y_test) = train_test_split(X_train, Y_train, test_size=0.2, random_state=101)
print('Train set:', x_train.shape, y_train.shape)
print('Test set:', x_test.shape, y_test.shape)
param_grid = {'n_estimators': [5, 10, 15, 20], 'max_depth': [2, 5, 7, 9]}
clf = XGBRegressor(random_state=42)