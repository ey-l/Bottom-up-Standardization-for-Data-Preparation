import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
_input5 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
_input0 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
_input2 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
plt.figure(figsize=(10, 4))
plt.xlim(_input0.item_cnt_day.min(), _input0.item_cnt_day.max() * 1.1)
sns.boxplot(x=_input0.item_cnt_day)
plt.figure(figsize=(10, 4))
plt.xlim(_input0.item_price.min(), _input0.item_price.max() * 1.1)
sns.boxplot(x=_input0.item_price)
_input0 = _input0.drop(_input0[_input0['item_cnt_day'] < 0].index, axis=0, inplace=False)
_input0 = _input0.drop_duplicates(subset=['date', 'date_block_num', 'shop_id', 'item_id', 'item_cnt_day'], inplace=False)
_input0 = _input0[_input0.item_cnt_day < 1200]
_input0 = _input0[_input0.item_price < 100000]
plt.figure(figsize=(10, 4))
plt.xlim(_input0.item_cnt_day.min() - 50, _input0.item_cnt_day.max() * 1.1)
sns.boxplot(x=_input0.item_cnt_day)
plt.figure(figsize=(10, 4))
plt.xlim(_input0.item_price.min() - 500, _input0.item_price.max() * 1.1)
sns.boxplot(x=_input0.item_price)
agg_df = _input0.groupby(['date_block_num', 'shop_id', 'item_id'])['item_cnt_day'].agg('sum').reset_index()
agg_df.columns = ['date_block_num', 'shop_id', 'item_id', 'item_cnt_day']
agg_df['item_cnt_day'] = agg_df['item_cnt_day'].clip(0, 20, inplace=False)
features = agg_df.iloc[:, :-1]
itemCount = agg_df.iloc[:, -1:]
(x_train, x_test, y_train, y_test) = train_test_split(features, itemCount, test_size=0.2, random_state=1)
rf = RandomForestRegressor(n_estimators=50, random_state=1, verbose=1)