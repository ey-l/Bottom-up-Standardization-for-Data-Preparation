import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from pandas import Series

import pmdarima as pm

def test_stationarity(timeseries):
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for (key, value) in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)
sales = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
sales['date'] = sales['date'].apply(lambda x: datetime.datetime.strptime(x, '%d.%m.%Y'))
sales
pairs = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
pairs
sales.shape
sales.info()
num_sales = sales.select_dtypes(include=['int64', 'float64'])
num_sales.describe()

def missing_ratio(s):
    return s.isna().mean() * 100

def nums_diff_values(s):
    return s.dropna().nunique()

def diff_vals_ratio(s):
    s = s.dropna()
    return (s.value_counts() / len(s) * 100).to_dict()
num_sales.agg([missing_ratio, nums_diff_values, diff_vals_ratio])
num_sales.hist(figsize=(10, 10), bins=200)
sns.heatmap(num_sales.corr(), annot=True)

sales.copy().set_index('date').item_cnt_day.resample('M').sum().plot()

sales.copy().set_index('date').item_cnt_day.resample('M').mean().plot()

total_sales = sales.groupby(['date_block_num'])['item_cnt_day'].sum().astype('float')
plt.title('Total Sales of the company')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.plot(total_sales)
plt.plot(total_sales.rolling(window=12, center=False).mean(), label='Rolling Mean')
plt.plot(total_sales.rolling(window=12, center=False).std(), label='Rolling Standard Deviation')
plt.legend()
res = sm.tsa.seasonal_decompose(total_sales.values, period=12, model='additive').plot()
test_stationarity(total_sales)
sales['shop_id'] = sales['shop_id'].replace({0: 57, 1: 58, 11: 10, 40: 39})
sales = sales.loc[sales.shop_id.isin(pairs['shop_id'].unique()), :]
sales = sales[(sales['item_price'] > 0) & (sales['item_price'] < 50000)]
sales = sales[(sales['item_cnt_day'] > 0) & (sales['item_cnt_day'] < 1000)]
data_df = sales.groupby(['date_block_num'])['date', 'item_cnt_day'].agg({'date': 'min', 'item_cnt_day': 'sum'})
data_df.rename(columns={'item_cnt_day': 'item_cnt_month'}, inplace=True)
data_df.set_index(['date'], inplace=True)
data_df.head()
plt.plot(data_df)
plt.tight_layout()

n = len(data_df)
train_test_ratio = 1.0
train_sr = data_df['item_cnt_month'][:int(n * train_test_ratio)]
test_sr = data_df['item_cnt_month'][int(n * train_test_ratio):]
print(f'Train/Test: {len(train_sr)}/{len(test_sr)}')
sarima_model = pm.auto_arima(y=train_sr, stationary=False, seasonal=True, test='kpss', seasonal_test='ocsb', start_p=1, max_p=4, d=None, start_q=1, max_q=4, start_P=1, max_P=4, D=None, start_Q=1, max_Q=4, m=12, trace=True, stepwise=True)
print(sarima_model.summary())
(predicted_values, confint) = sarima_model.predict(n_periods=3 * 12, return_conf_int=True)
confint_df = pd.DataFrame(confint)
date_index = pd.date_range(start=train_sr.index[-1], periods=3 * 12, freq='MS')
predicted_df = pd.DataFrame({'value': predicted_values}, index=date_index)
predicted_df
plt.plot(data_df, label='Actual data')
plt.plot(predicted_df, color='orange', label='Predicted data')
plt.fill_between(date_index, confint_df[0], confint_df[1], color='grey', alpha=0.3, label='Confidence Intervals Area')
plt.legend()

total_sales_last_month = data_df.values[-1][0]
total_sales_next_month = predicted_values[0]
print(f'Last month: {total_sales_last_month}\nNext month: {total_sales_next_month}')
grouped_data_df = sales.groupby(['shop_id', 'item_id'])['date', 'item_cnt_day'].agg({'item_cnt_day': 'sum'})
grouped_data_df.rename(columns={'item_cnt_day': 'item_cnt_all'}, inplace=True)
grouped_data_df = grouped_data_df.reset_index()
pairs['item_cnt_month'] = total_sales_next_month / len(pairs) * (len(pairs) / len(grouped_data_df))
results_df = pairs.drop(['shop_id', 'item_id'], axis=1)

results_df