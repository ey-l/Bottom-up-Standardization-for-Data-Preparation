
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:.1f}'.format
np.set_printoptions(suppress=True)
sales = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv', parse_dates=['date'])
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
sales.head()
sales_train = sales.join(other=shops, on='shop_id', how='inner', rsuffix='_').join(items, on='item_id', how='inner', rsuffix='_').join(categories, on='item_category_id', how='inner', rsuffix='_')
sales_train.drop(['shop_id_', 'item_id_', 'item_category_id_'], axis=1, inplace=True)
sales_train.head()
sales_train.info()
sales_train.describe()
len(sales_train[sales_train['item_cnt_day'] < 0])
sales_train = sales_train[(sales_train['item_cnt_day'] > 0) & (sales_train['item_price'] > 0)]
sales_train.isna().sum()
plt.scatter(sales_train.index, sales_train['item_cnt_day'])

p = 99
percentile = np.percentile(sales_train['item_cnt_day'], p)
print(f'The item_cnt_day {p}th percentile is equal to {percentile}')
p = 99.9
percentile = np.percentile(sales_train['item_cnt_day'], p)
print(f'The item_cnt_day {p}th percentile is equal to {percentile}')
outliers_num = len(sales_train[sales_train['item_cnt_day'] > percentile])
sales_train = sales_train[sales_train['item_cnt_day'] < percentile]
print(f'We removed {outliers_num} outliers from the data')
print(sales_train['item_id'].nunique())
print(sales_train['shop_id'].nunique())
print(len(test))
print(f"Min date from train set: {sales_train['date'].min().date()}")
print(f"Max date from train set: {sales_train['date'].max().date()}")
sales_train['month'] = sales_train['date'].dt.month
sales_train['year'] = sales_train['date'].dt.year
N = 15
items_total_sold = sales_train.groupby('item_id').sum()
items_total_sold.reset_index(inplace=True)
idxs = items_total_sold['item_cnt_day'].values.argsort()[::-1][0:N]
temp = items_total_sold['item_cnt_day'].to_numpy()
max_sold = [temp[idx] for idx in idxs]
item_ids = items_total_sold.loc[idxs, 'item_id'].values
item_names = items.loc[item_ids, 'item_name']
(fig, ax) = plt.subplots(figsize=(16, 12))
barplot = sns.barplot(x=max_sold, y=item_names)
for p in barplot.patches:
    barplot.text(p.get_width(), p.get_y() + 0.55 * p.get_height(), '{:1.0f}'.format(p.get_width()), ha='center', va='center')
barplot.set(xlabel='Number of items sold', ylabel='Items', title=f'Top {N} items sold')

sales_train['transaction_price'] = sales_train['item_cnt_day'] * sales_train['item_price']
total_revenue_shops = sales_train.groupby('shop_id').agg({'transaction_price': ['sum']})
total_shops = total_revenue_shops['transaction_price']['sum'].sort_values(ascending=False)
index = total_shops.index
sns.set_theme(context='notebook', style='whitegrid', font_scale=1.3)
(fig, ax) = plt.subplots(figsize=(20, 10))
barplot = sns.barplot(x=index, y=total_shops, order=index)
barplot.set(xlabel='Shop id', ylabel='Total revenues', title="Shops's rank in term of revenues")
plt.ticklabel_format(style='plain', axis='y')

avg_total_sales_day = sales_train.groupby('date').agg({'item_cnt_day': ['sum']}).mean().values[0]
print(f'In average, total sales per day is about {round(avg_total_sales_day)} items sold.')
total_sales_day = sales_train.groupby('date').agg({'item_cnt_day': ['sum']})['item_cnt_day']['sum']
(fig, ax) = plt.subplots(figsize=(20, 10))
lineplot = sns.lineplot(x=total_sales_day.index, y=total_sales_day)
lineplot.set(xlabel='Date', ylabel='Total sales', title='Total day sales through time among all shops')

avg_total_sales_month = sales_train.groupby(['year', 'month']).agg({'item_cnt_day': ['sum']}).mean().values[0]
print(f'In average, total sales per month is about {round(avg_total_sales_month)} items sold.')
y = sales_train.groupby(['year', 'month']).agg({'item_cnt_day': ['sum']})['item_cnt_day']['sum'].values
date = []
years = ['2013', '2014', '2015']
for year in years:
    for month in range(1, 13):
        if month < 10:
            date.append(year + '-0' + str(month))
        else:
            date.append(year + '-' + str(month))
(fig, ax) = plt.subplots(figsize=(20, 10))
lineplot = sns.lineplot(x=date, y=y)
lineplot.set(xlabel='Date', ylabel='Total sales', title='Total month sales through time among all shops')
lineplot.set_xticklabels(date, rotation=50)

sales_train.groupby('date').agg({'item_cnt_day': ['sum']}).tail(12)
N = 10
total_categories_sold = sales_train.groupby('item_category_id').sum()
items_sold_by_category = total_categories_sold['item_cnt_day'].nlargest(n=N)
idxs = items_sold_by_category.index
categories_names = categories.loc[idxs, 'item_category_name']
(fig, ax) = plt.subplots(figsize=(16, 8))
barplot = sns.barplot(x=categories_names, y=items_sold_by_category)
barplot.set(xlabel='Category name', ylabel='Total items sold', title=f'Top {N} most popular categories')
for label in barplot.get_xticklabels():
    label.set_rotation(50)
for p in barplot.patches:
    barplot.annotate(format(p.get_height(), '1.0f'), (p.get_x() + p.get_width() / 2.0, p.get_height()), ha='center', va='center', xytext=(0, 7), textcoords='offset points')
plt.ticklabel_format(style='plain', axis='y')

data_sales = sales_train.groupby(['date_block_num', 'shop_id', 'item_id'], as_index=False)['item_cnt_day'].sum()
data_sales.rename(columns={'item_cnt_day': 'item_cnt_month'}, inplace=True)
data_sales = pd.pivot_table(data_sales, index=['shop_id', 'item_id'], columns=['date_block_num'], fill_value=0)
data_sales
df_sales = data_sales.merge(test, how='right', on=['shop_id', 'item_id'])
per = round(df_sales.isna().sum()[2] / len(df_sales), 2)
df_sales
print(f'The percentage of (shop_id, item_id) combinations with NaN values is {per}')
df_sales.fillna(0, inplace=True)
df_sales.drop(['shop_id', 'item_id', 'ID'], axis=1, inplace=True)
df_sales.head()
from sklearn.model_selection import train_test_split
(X, y) = (df_sales.drop(labels=[('item_cnt_month', 33)], axis=1).values, df_sales.values[:, -1])
X_test = df_sales.drop(labels=[('item_cnt_month', 0)], axis=1).values
X = X.reshape(X.shape[0], X.shape[1], 1)
y = y.reshape(y.shape[0], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
(X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=0)
print('X_train shape : ', X_train.shape)
print('y_train shape : ', y_train.shape)
print('X_val shape : ', X_val.shape)
print('y_val shape : ', y_val.shape)
print('X_test shape : ', X_test.shape)
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
tf.keras.backend.set_floatx('float64')
early_stopping = EarlyStopping(monitor='val_loss', patience=2, mode='auto')

def plot_train_val_curves(hist):
    (fig, (ax1, ax2)) = plt.subplots(2, figsize=(15, 8), sharex=True)
    fig.suptitle('Loss and MSE train and validation curves')
    ax1.plot(hist.history['loss'], label='train')
    ax1.plot(hist.history['val_loss'], label='validation')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax2.plot(hist.history['mean_squared_error'], label='train')
    ax2.plot(hist.history['val_mean_squared_error'], label='validation')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('MSE')
    ax2.legend()


def LSTM_model(shape, units=[64, 64], dropout=0.3):
    model = Sequential()
    model.add(Input(shape=shape, dtype='float64'))
    model.add(LSTM(units=units[0], activation='tanh', return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(units=units[1], activation='tanh'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation=None, kernel_initializer=tf.initializers.zeros()))
    return model
lstm_model = LSTM_model(shape=(X_train.shape[1], X_train.shape[2]), units=[128, 128], dropout=0.4)
lstm_model.compile(optimizer='Adam', loss='mse', metrics=['mean_squared_error'])
lstm_model.summary()