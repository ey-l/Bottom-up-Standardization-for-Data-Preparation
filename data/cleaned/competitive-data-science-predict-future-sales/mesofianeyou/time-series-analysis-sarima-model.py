
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud
import datetime
import matplotlib as mpl
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima.arima import ARIMA
from pmdarima.arima import auto_arima
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
item_categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
sales_train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
styles = [dict(selector='caption', props=[('font-size', '120%'), ('font-weight', 'bold'), ('background-color', 'cyan'), ('color', 'black'), ('text-align', 'center')])]

item_categories.info()

items.info()

shops.info()

sales_train.info()
items_merged = pd.merge(item_categories, items, how='inner')
sales_train_merged = pd.merge(sales_train, shops, on='shop_id')
sales_train_merged = pd.merge(sales_train_merged, items_merged, on='item_id')
sales_train_merged.head().style.background_gradient(cmap='Blues').set_properties(**{'font-family': 'Segoe UI'})
text1 = ' '.join((title for title in sales_train_merged.item_name))
word_cloud1 = WordCloud(collocations=False, background_color='white', width=2048, height=1080).generate(text1)
word_cloud1.to_file('got.png')
plt.figure(figsize=[15, 10])
plt.imshow(word_cloud1, interpolation='bilinear')
plt.axis('off')

result = items_merged['item_category_name'].value_counts().sort_values(ascending=False)[0:20]
result.plot(kind='bar', figsize=(12, 5), width=0.8, color=sns.color_palette('Spectral', 9))
plt.title('Number of items in each category')
plt.ylabel('Number of items')

plt.figure(figsize=(12, 5))
plt.title('top categories')
plt.ylabel('item_cnt_day')
sales_train_merged.groupby('item_category_name')['item_cnt_day'].sum().sort_values(ascending=False)[0:15].plot(kind='line', marker='*', color='red', ms=10)
sales_train_merged.groupby('item_category_name')['item_cnt_day'].sum().sort_values(ascending=False)[0:15].plot(kind='bar', color=sns.color_palette('inferno_r', 7))

plt.figure(figsize=(12, 5))
plt.title('top shops')
plt.ylabel('item_cnt_day')
sales_train_merged.groupby('shop_name')['item_cnt_day'].sum().sort_values(ascending=False)[0:10].plot(kind='line', marker='*', color='red', ms=10)
sales_train_merged.groupby('shop_name')['item_cnt_day'].sum().sort_values(ascending=False)[0:10].plot(kind='bar', color=sns.color_palette('viridis', 100))

plt.figure(figsize=(12, 5))
plt.title('top categories')
plt.ylabel('item_price')
sales_train_merged.groupby('item_category_name')['item_price'].mean().sort_values(ascending=True)[70:84].plot(kind='line', marker='*', color='red', ms=10)
sales_train_merged.groupby('item_category_name')['item_price'].mean().sort_values(ascending=True)[70:84].plot(kind='bar', color=sns.color_palette('inferno_r', 7))

sales_train['date'] = sales_train['date'].apply(lambda x: datetime.datetime.strptime(x, '%d.%m.%Y'))
df_train = sales_train.copy()
sales_train = sales_train.set_index('date')
plt.style.use('ggplot')
ax2 = sales_train['item_cnt_day'].plot(figsize=(16, 6), color='blue')
ax2.set_title('item_cnt_day plot')
ax2.set_xlabel('Date')
ax2.set_ylabel('item_cnt_day')

(fig, ax) = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
sales_train.loc['2013', 'item_cnt_day'].resample('M').plot(ax=ax[0, 0])
sales_train.loc['2014', 'item_cnt_day'].resample('M').plot(ax=ax[0, 1])
sales_train.loc['2015', 'item_cnt_day'].resample('M').plot(ax=ax[0, 2])
sales_train.loc['2013', 'item_cnt_day'].resample('M').sum().plot(ax=ax[1, 0], color='blue')
sales_train.loc['2014', 'item_cnt_day'].resample('M').sum().plot(ax=ax[1, 1], color='red')
sales_train.loc['2015', 'item_cnt_day'].resample('M').sum().plot(ax=ax[1, 2], color='green')
plt.suptitle('item_cnt_day from 2013 to 2015', fontsize=20)
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.92, wspace=0.4, hspace=0.4)
fig.tight_layout()

df_train['month'] = df_train['date'].dt.to_period('M')
df_train['month'] = df_train['month'].astype(str)
dff2 = df_train.copy()
df_train['month'] = pd.to_datetime(df_train['month'])
dff_train = df_train.groupby(['month']).agg({'item_cnt_day': 'sum'})
dff_train['month'] = dff_train.index
dff_train.drop(['month'], axis=1, inplace=True)
dff_train.rename(columns={'item_cnt_day': 'item_cnt_month'}, inplace=True)
dff2 = dff2.groupby(['month']).agg({'item_cnt_day': 'sum'})
dff2['month'] = dff2.index
dff2.rename(columns={'item_cnt_day': 'item_cnt_month'}, inplace=True)
data = dff2['item_cnt_month'].values
doublediff = np.diff(np.sign(np.diff(data)))
peak_locations = np.where(doublediff == -2)[0] + 1
doublediff2 = np.diff(np.sign(np.diff(-1 * data)))
trough_locations = np.where(doublediff2 == -2)[0] + 1
plt.figure(figsize=(17, 7))
plt.plot('month', 'item_cnt_month', data=dff2, color='blue', label='Air Traffic')
plt.scatter(dff2.month[peak_locations], dff2.item_cnt_month[peak_locations], marker=mpl.markers.CARETUPBASE, color='green', s=100, label='Peaks')
plt.scatter(dff2.month[trough_locations], dff2.item_cnt_month[trough_locations], marker=mpl.markers.CARETDOWNBASE, color='red', s=100, label='Troughs')
for (t, p) in zip(trough_locations[1::5], peak_locations[::3]):
    plt.text(dff2.month[p], dff2.item_cnt_month[p] + 15, dff2.month[p], horizontalalignment='center', color='darkgreen')
    plt.text(dff2.month[t], dff2.item_cnt_month[t] - 35, dff2.month[t], horizontalalignment='center', color='darkred')
xtick_location = dff2.index.tolist()[::6]
xtick_labels = dff2.month.tolist()[::6]
plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=90, fontsize=12, alpha=0.7)
plt.title('item_cnt_month (2013 - 2015)', fontsize=22)
plt.yticks(fontsize=12, alpha=0.7)
plt.gca().spines['top'].set_alpha(0.0)
plt.gca().spines['bottom'].set_alpha(0.3)
plt.gca().spines['right'].set_alpha(0.0)
plt.gca().spines['left'].set_alpha(0.3)
plt.legend(loc='upper left')
plt.grid(axis='y', alpha=0.3)

decomposition = seasonal_decompose(dff_train, model='additive')
trend = decomposition.trend
seasonal = decomposition.seasonal
sales_decomposed = pd.DataFrame(np.c_[trend, seasonal], index=dff_train.index, columns=['trend', 'seasonal'])
ax = sales_decomposed.plot(figsize=(12, 6), fontsize=15)
ax.set_xlabel('Date', fontsize=15)
plt.legend(fontsize=15)

seasonal_decompose(dff_train, model='additive').plot().set_size_inches(10, 8)
seasonal_decompose(dff_train, model='multiplicative').plot().set_size_inches(10, 8)
result = adfuller(dff_train)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for (key, value) in result[4].items():
    print('\t%s: %.3f' % (key, value))
sd_dff_train = dff_train - dff_train.shift(12)
sd_dff_train = sd_dff_train.dropna()
sd_dff_train.plot(figsize=(12, 6))

result = adfuller(sd_dff_train)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
(fig, axes) = plt.subplots(nrows=1, ncols=2, figsize=(16, 5))
plot_acf(dff_train, lags=10, ax=axes[0])
plot_pacf(dff_train, lags=10, ax=axes[1])

model = auto_arima(dff_train, start_p=0, start_q=0, D=1, m=12, seasonal=True, test='adf', trace=True, alpha=0.05, information_criterion='aic', suppress_warnings=True, stepwise=True)
model.summary()
model.plot_diagnostics(figsize=(14, 10))

(prediction, confint) = model.predict(n_periods=6, return_conf_int=True)
period_index = pd.period_range(start=dff_train.index[-1], periods=6, freq='M')
forecast = pd.DataFrame({'Predicted item_cnt_month': prediction.round(2)}, index=period_index)
forecast
cf = pd.DataFrame(confint)
prediction_series = pd.Series(prediction, index=period_index)
plt.figure(figsize=(15, 5))
plt.plot(dff_train, color='red', label='Actual')
plt.plot(prediction_series, color='orange', label='Predicted')
plt.fill_between(prediction_series.index, cf[0], cf[1], color='grey', alpha=0.2, label='Confidence Intervals Area')
plt.legend()

train_dfff = df_train.groupby(['shop_id', 'item_id'])['date', 'item_cnt_day'].agg({'item_cnt_day': 'sum'})
train_dfff = train_dfff.reset_index()
print(train_dfff)
test['item_cnt_month'] = prediction[0].round(2) * len(test) / len(train_dfff) / len(test)
submission = test.drop(['shop_id', 'item_id'], axis=1)
print(submission)
