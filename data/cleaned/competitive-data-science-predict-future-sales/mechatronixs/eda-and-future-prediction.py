import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')
import seaborn as sns
import sys
import itertools
import gc
import datetime
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import csv
from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
cats = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')

def check_df(dataframe, head=5):
    print('#### Shape #### ')
    print(dataframe.shape)
    print('### Types ###')
    print(dataframe.dtypes)
    print('### Head ###')
    print(dataframe.head(head))
    print('### Tail ###')
    print(dataframe.tail(head))
    print('### NA ###')
    print(dataframe.isnull().sum())
    print('### Quantiles ###')
    print(dataframe.describe([0, 0.05, 0.5, 0.95, 0.99, 1]).T)
check_df(train)
train['date'] = pd.to_datetime(train['date'], format='%d.%m.%Y')
train['date']
train = train[(train['item_price'] > 0) & (train['item_price'] < 10000)]
train = train[(train['item_cnt_day'] > 0) & (train['item_cnt_day'] < 1000)]
train.head()
train['month_year'] = train['date'].dt.to_period('M')
train.head()
grouped_df = train.groupby(['month_year'])['month_year', 'item_cnt_day'].agg({'item_cnt_day': 'sum'})
grouped_df = grouped_df.reset_index()
grouped_df.set_index(['month_year'], inplace=True)
grouped_df.rename(columns={'item_cnt_day': 'item_cnt_month'}, inplace=True)
grouped_df.head(10)

import pmdarima as pm
from pmdarima.arima import auto_arima
model = auto_arima(y=grouped_df, seasonal=True, start_p=1, max_p=5, start_q=1, max_q=5, d=None, start_P=1, max_P=5, start_Q=1, max_Q=5, D=None, m=12)
print(model.summary())
(prediction, confint) = model.predict(n_periods=12, return_conf_int=True)
confint_df = pd.DataFrame(confint)
prediction
period_index = pd.period_range(start=grouped_df.index[-1], periods=12, freq='M')
predicted_df = pd.DataFrame({'value': prediction}, index=period_index)
predicted_df
plt.figure(figsize=(10, 8))
plt.plot(grouped_df.to_timestamp(), label='Actual data')
plt.plot(predicted_df.to_timestamp(), color='orange', label='Predicted data')
plt.fill_between(period_index.to_timestamp(), confint_df[0], confint_df[1], color='grey', alpha=0.2, label='Confidence Intervals Area')
plt.legend()

print(f'sales last month: {grouped_df.values[-1][0]}')
print(f'sales next month: {prediction[0]}')
group_pair_train = train.groupby(['shop_id', 'item_id'])['date', 'item_cnt_day'].agg({'item_cnt_day': 'sum'})
group_pair_train = group_pair_train.reset_index()
group_pair_train.head(10)
test['item_cnt_month'] = prediction[0] * len(test) / len(group_pair_train) / len(test)
submission = test.drop(['shop_id', 'item_id'], axis=1)
