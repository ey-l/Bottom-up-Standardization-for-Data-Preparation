
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import statsmodels.tsa.api as tsa
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import statsmodels.api as sm
import numpy as np
df_train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
df_train.head()
df_test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
df_test.head()
print('Original length', len(df_train))
df_train = df_train[df_train['shop_id'].isin(df_test['shop_id'])]
print('Removed shop length', len(df_train))
df_train = df_train[df_train['item_id'].isin(df_test['item_id'])]
print('Removed items length', len(df_train))
mon_sales = df_train.groupby(['date_block_num']).agg(item_cnt_mon=('item_cnt_day', 'sum'))
mon_sales.head()
sm.stats.acorr_ljungbox(mon_sales['item_cnt_mon'])
sns.set_theme()

def plot_ts(x, lags=10, alpha=0.05):
    plt.figure(figsize=(10, 7.5))
    ts = pd.Series(x)
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))
    ts_ax.set_title('original time series')
    ts.plot(ax=ts_ax)
    tsa.graphics.plot_acf(ts, ax=acf_ax, lags=lags, alpha=alpha)
    tsa.graphics.plot_pacf(ts, ax=pacf_ax, lags=lags, alpha=alpha)
    plt.tight_layout()

tsa.adfuller(mon_sales['item_cnt_mon'])[1]
tsa.adfuller(mon_sales['item_cnt_mon'].diff()[1:])[1]
plot_ts(mon_sales['item_cnt_mon'].diff()[1:])
model_2 = ARIMA(mon_sales['item_cnt_mon'], order=(2, 1, 0))