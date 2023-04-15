import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
sample_submission = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
item_categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
print(train.shape)
print(test.shape)
print(items.shape)
print(item_categories.shape)
print(shops.shape)
train.head()
test.head()
items.head()
item_categories.head()
shops.head()
train.item_cnt_day.plot()
plt.title('Number of products sold per day')
l = train.select_dtypes(include=['float64', 'int64'])
l.hist(figsize=(16, 16), bins=50, xlabelsize=8, ylabelsize=8)
unique_dates = pd.DataFrame({'date': train['date'].drop_duplicates()})
unique_dates['date_parsed'] = pd.to_datetime(unique_dates.date, format='%d.%m.%Y')
unique_dates['day'] = unique_dates['date_parsed'].apply(lambda d: d.day)
unique_dates['month'] = unique_dates['date_parsed'].apply(lambda d: d.month)
unique_dates['year'] = unique_dates['date_parsed'].apply(lambda d: d.year)
datess = train.merge(unique_dates, on='date').sort_values('date_parsed')
data = datess.groupby(['year', 'month']).agg({'item_cnt_day': np.sum}).reset_index().pivot(index='month', columns='year', values='item_cnt_day')
data.plot(figsize=(12, 8))
target = train.item_cnt_day
train = train.drop(['item_price', 'item_cnt_day', 'date', 'date_block_num'], axis=1).select_dtypes(exclude=['object'])
(train_X, test_X, train_y, test_y) = train_test_split(train, target, test_size=0.25)
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)