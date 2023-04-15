import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import IsolationForest
import seaborn as sns
import matplotlib.pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
submission_sample = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
(train.head(), test.head(), submission_sample.head())
plt.figure(figsize=(10, 6))
sns.set_style('whitegrid')
sns.boxplot(x='item_price', data=train)
train = train[train.item_price < 70000]
train.shape
plt.figure(figsize=(10, 6))
sns.set_style('whitegrid')
sns.boxplot(x='item_cnt_day', data=train)
train = train[train.item_cnt_day < 1000]
agg = train.groupby(['date_block_num', 'shop_id', 'item_id']).item_cnt_day.sum().reset_index(name='item_cnt_monthly')
agg
agg2 = agg.groupby(['shop_id', 'item_id']).item_cnt_monthly.mean().reset_index(name='item_cnt_month')
agg2
agg2.isna().sum()
agg2.corr()
X = pd.DataFrame(agg2['item_id'])
y = agg2['item_cnt_month']
X.head()
(Xtrain, Xval, ytrain, yval) = train_test_split(X, y, train_size=0.8)
Xtest = test[['item_id']]
model = XGBRegressor()