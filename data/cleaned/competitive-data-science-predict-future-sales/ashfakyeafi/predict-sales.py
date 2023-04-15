
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima
df_train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
df_test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
(df_train.shape, df_test.shape)
(df_train.dtypes, df_test.columns)
df_train['date'] = df_train['date'].apply(lambda x: datetime.datetime.strptime(x, '%d.%m.%Y'))
df_train['month'] = df_train['date'].dt.to_period('M')
df_train['month'] = df_train['month'].astype(str)
df_train['month'] = pd.to_datetime(df_train['month'])
df_train.dtypes
dff_train = df_train.groupby(['month']).agg({'item_cnt_day': 'sum'})
dff_train['month'] = dff_train.index
dff_train.rename(columns={'item_cnt_day': 'item_cnt_month'}, inplace=True)
print(dff_train.shape, dff_train.columns)
len(dff_train.drop(['month'], axis=1))
plt.figure(figsize=(9, 6))
plt.grid()
plt.plot(dff_train['item_cnt_month'])
plt.title('Monthly Sales of items')
plt.xlabel('Time')
plt.ylabel('Sales count')

pd.plotting.autocorrelation_plot(dff_train['item_cnt_month'])
print('Autocorrelation =', round(dff_train['item_cnt_month'].autocorr(), 4))
plot_acf(dff_train['item_cnt_month'])
plt.grid()
plot_pacf(dff_train['item_cnt_month'])
plt.grid()

seasonal_decompose(dff_train['item_cnt_month'], model='additive').plot().set_size_inches(10, 8)
seasonal_decompose(dff_train['item_cnt_month'], model='multiplicative').plot().set_size_inches(10, 8)

def adf_test(dataseries):
    adf = adfuller(dataseries)
    output = pd.Series(adf[0:3], index=['ADF Statistic', 'p-value', 'Lags'])
    for (key, value) in adf[4].items():
        output['Critical Value (%s)' % key] = value
    return print(output)
adf_test(dff_train['item_cnt_month'])

def order_parameters(training_data):
    search_params = auto_arima(training_data, start_p=0, start_q=0, m=12, seasonal=True, test='adf', d=None, trace=True, alpha=0.05, information_criterion='aic', suppress_warnings=True, stepwise=True)
    print('AIC = ', round(search_params.aic(), 2))
    return search_params
model = order_parameters(dff_train['item_cnt_month'])
print(model.summary())
(prediction, confint) = model.predict(n_periods=6, return_conf_int=True)
df_confint = pd.DataFrame(confint)
print(confint.round(2))
print(prediction.round(2))
period_index = pd.period_range(start=dff_train.index[-1], periods=6, freq='M')
df_predict = pd.DataFrame({'Predicted item_cnt_month': prediction.round(2)}, index=period_index)
print(df_predict)
df_predict.head(2)
plt.figure(figsize=(10, 6))
plt.plot(dff_train['item_cnt_month'], label='Actuals')
plt.plot(df_predict.to_timestamp(), color='orange', label='Predicted')
plt.fill_between(period_index.to_timestamp(), df_confint[0], df_confint[1], color='grey', alpha=0.25, label='Confidence Interval')
plt.legend(loc='lower left')
plt.title('Time-series Forecasting (SARIMA)')
plt.grid()

train_df_tuple = df_train.groupby(['shop_id', 'item_id'])['date', 'item_cnt_day'].agg({'item_cnt_day': 'sum'})
train_df_tuple = train_df_tuple.reset_index()
print(train_df_tuple)
df_test['item_cnt_month'] = prediction[0].round(2) * len(df_test) / len(train_df_tuple) / len(df_test)
submission = df_test.drop(['shop_id', 'item_id'], axis=1)
print(submission)
