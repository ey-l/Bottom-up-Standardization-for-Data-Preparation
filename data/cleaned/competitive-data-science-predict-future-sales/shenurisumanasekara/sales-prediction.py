import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import sklearn

from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from scipy import stats
import sklearn.metrics as sm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
item_set = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
item_category_set = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
train_set = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
shop_set = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
test_set = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
item_set.head()
item_set.shape
item_set['item_name'].nunique()
item_category_set.head()
item_category_set.shape
item_category_set['item_category_name'].nunique()
shop_set.head()
shop_set.shape
shop_set['shop_name'].nunique()
train_set.head(20)
item_price = train_set['item_price']
item_price.sort_values(ascending=False)
train_set.shape
train_set.info()
train_set.describe()
train_set.isnull().sum()
train_set.hist(figsize=(15, 15), bins=6)

train_set['item_cnt_day'].hist(range=[-1, 10], facecolor='green', align='mid')

sns.displot(train_set['item_cnt_day'])
train_set['item_cnt_day'].describe()
correlation_matrix = train_set.corr()
correlation_matrix['item_cnt_day'].sort_values(ascending=False)
correlation_num = 6
correlation_cols = correlation_matrix.nlargest(correlation_num, 'item_cnt_day')['item_cnt_day'].index
correlation_mat_sales = np.corrcoef(train_set[correlation_cols].values.T)
sns.set(font_scale=1.25)
(f, ax) = plt.subplots(figsize=(12, 9))
hm = sns.heatmap(correlation_mat_sales, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 7}, yticklabels=correlation_cols.values, xticklabels=correlation_cols.values)

train_set = train_set.drop(columns=['date', 'date_block_num', 'item_price'])
train_set.head()
y = train_set['item_cnt_day']
x = train_set.drop(columns=['item_cnt_day'])
print(len(x.columns))
(X_train, X_test, Y_train, Y_test) = train_test_split(x, y, test_size=0.3, random_state=60, shuffle=True)
print(len(X_train))
print(len(X_test))
linear_model = LinearRegression()