import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import plot_importance
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
X = train.copy()
X = X[X.item_price < 100000]
X = X[X.item_cnt_day < 1001]
X.dropna(axis=0, subset=['item_cnt_day'], inplace=True)
X['item_cnt_day'][X['item_cnt_day'] < 0] = 0
X['year'] = pd.DatetimeIndex(X['date'], dayfirst=True).year - 2012
X['month'] = pd.DatetimeIndex(X['date'], dayfirst=True).month
"\nX['target_month'] = X['date'].str.contains(pat = '.10').astype(int)\ntest['target_month'] = [1]* len(test['shop_id'])\n"
X.drop(['date'], axis=1, inplace=True)
X.head()
X.drop(['item_price'], axis=1, inplace=True)
X = X.groupby(['date_block_num', 'shop_id', 'item_id']).mean().reset_index()
a = (np.mean(np.array([h.get_height() for h in sns.distplot(X[X['month'] == 11]['item_id']).patches])) * 100000000).round(decimals=0)
plt.close()
(f, axes) = plt.subplots(2, 2)
plt.subplots_adjust(right=1.8, top=1.8)
sns.distplot(X['item_id'], kde=False, ax=axes[0, 0])
axes[0, 0].axhline(a, ls='--', color='r')
sns.distplot(test['item_id'], kde=False, ax=axes[0, 1])
axes[0, 1].axhline(a, ls='--', color='r')
sns.distplot(X['shop_id'], kde=False, color='r', ax=axes[1, 0])
axes[1, 0].axhline(test[test['shop_id'] == 50]['shop_id'].count(), ls='--')
sns.distplot(test['shop_id'], kde=False, color='r', ax=axes[1, 1])
axes[1, 1].axhline(test[test['shop_id'] == 50]['shop_id'].count(), ls='--')
new_X = X[(X['month'] == 11) | (X['month'] == 10) | (X['month'] == 12)]
dr = new_X[(new_X['item_id'] > 1000) & (new_X['item_id'] < 8000)].sample(frac=0.3)
new_X = new_X.drop(dr.index)
(f, axes) = plt.subplots(2, 2)
plt.subplots_adjust(right=1.8, top=1.8)
sns.distplot(new_X['item_id'], kde=False, ax=axes[0, 0])
axes[0, 0].axhline(a, ls='--', color='r')
sns.distplot(test['item_id'], kde=False, ax=axes[0, 1])
axes[0, 1].axhline(a, ls='--', color='r')
sns.distplot(new_X['item_id'], kde=False, color='r', ax=axes[1, 0])
axes[1, 0].axhline(test[test['shop_id'] == 50]['shop_id'].count(), ls='--')
sns.distplot(test['shop_id'], kde=False, color='r', ax=axes[1, 1])
axes[1, 1].axhline(test[test['shop_id'] == 50]['shop_id'].count(), ls='--')
print(len(X[X['month'] == 11]['item_id']), len(test['item_id']))
X.drop(['year', 'month'], axis=1, inplace=True)
new_X.drop(['year', 'month'], axis=1, inplace=True)
undo = X
X = new_X
test_list = pd.DataFrame()
print('Not fulfilled:')
for i in range(60):
    test_samples = test[test['shop_id'] == i + 1]['shop_id'].count()
    if test_samples != 0:
        try:
            X_test_samples = X[X['shop_id'] == i + 1].sample(n=test_samples)
            X = X.drop(X_test_samples.index)
            test_list = test_list.append(X_test_samples, ignore_index=True)
        except:
            print('shop:', i + 1, 'samples:', test_samples, 'X samples:', len(X[X['shop_id'] == i + 1]))
            print(len(X[X['shop_id'] == i + 1]), 'instead')
            X_test_samples = X[X['shop_id'] == i + 1].sample(n=len(X[X['shop_id'] == i + 1]))
            X = X.drop(X_test_samples.index)
            test_list = test_list.append(X_test_samples, ignore_index=True)
(f, axes) = plt.subplots(2, 2)
plt.subplots_adjust(right=1.8, top=1.8)
sns.distplot(test_list['item_id'], kde=False, ax=axes[0, 0])
axes[0, 0].axhline(a, ls='--', color='r')
sns.distplot(test['item_id'], kde=False, ax=axes[0, 1])
sns.distplot(test_list['shop_id'], kde=False, color='r', ax=axes[1, 0])
sns.distplot(test['shop_id'], kde=False, color='r', ax=axes[1, 1])
undo = undo.drop(X.index)
X = undo
y_train = X.item_cnt_day
X.drop(['item_cnt_day'], axis=1, inplace=True)
X_train = X
y_valid = test_list.item_cnt_day
test_list.drop(['item_cnt_day'], axis=1, inplace=True)
X_valid = test_list

my_model2 = XGBRegressor(n_estimators=1000, min_child_weight=200, eta=0.5)