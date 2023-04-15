import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
submission = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
sales = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
plt.figure(figsize=(10, 4))
plt.xlim(sales.item_cnt_day.min(), sales.item_cnt_day.max() * 1.1)
sns.boxplot(x=sales.item_cnt_day)
plt.figure(figsize=(10, 4))
plt.xlim(sales.item_price.min(), sales.item_price.max() * 1.1)
sns.boxplot(x=sales.item_price)
sales.drop(sales[sales['item_cnt_day'] < 0].index, axis=0, inplace=True)
sales.drop_duplicates(subset=['date', 'date_block_num', 'shop_id', 'item_id', 'item_cnt_day'], inplace=True)
sales = sales[sales.item_cnt_day < 1200]
sales = sales[sales.item_price < 100000]
plt.figure(figsize=(10, 4))
plt.xlim(sales.item_cnt_day.min() - 50, sales.item_cnt_day.max() * 1.1)
sns.boxplot(x=sales.item_cnt_day)
plt.figure(figsize=(10, 4))
plt.xlim(sales.item_price.min() - 500, sales.item_price.max() * 1.1)
sns.boxplot(x=sales.item_price)
agg_df = sales.groupby(['date_block_num', 'shop_id', 'item_id'])['item_cnt_day'].agg('sum').reset_index()
agg_df.columns = ['date_block_num', 'shop_id', 'item_id', 'item_cnt_day']
agg_df['item_cnt_day'].clip(0, 20, inplace=True)
features = agg_df.iloc[:, :-1]
itemCount = agg_df.iloc[:, -1:]
(x_train, x_test, y_train, y_test) = train_test_split(features, itemCount, test_size=0.2, random_state=1)
rf = RandomForestRegressor(n_estimators=50, random_state=1, verbose=1)