import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
sample_submission = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
item_categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
sales_train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
print(items.info())
print('Items : \n\t' + '\n\t'.join(list(items)))
print('ItemCatagories : \n\t' + '\n\t'.join(list(item_categories.columns.values)))
print('Shops : \n\t' + '\n\t'.join(shops.columns.tolist()))
print('SalesTrain : \n\t' + '\n\t'.join(sales_train.columns.tolist()))
print('Test : \n\t' + '\n\t'.join(list(test)))
print('OutPut : \n\t' + '\n\t'.join(list(sample_submission)))
sales_train.info()
print('Items')
print(items.head(2))
print('\nItem Catagerios')
print(item_categories.tail(2))
print('\nShops')
print(shops.sample(n=2))
print('\nTraining Data Set')
print(sales_train.sample(n=3, random_state=1))
print('\nTest Data Set')
print(test.sample(n=3, random_state=1))
from datetime import datetime
sales_train['year'] = pd.to_datetime(sales_train['date']).dt.strftime('%Y')
sales_train['month'] = sales_train.date.apply(lambda x: datetime.strptime(x, '%d.%m.%Y').strftime('%m'))
sales_train.head(2)
import matplotlib.pyplot as plt
import seaborn as sns

grouped = pd.DataFrame(sales_train.groupby(['year', 'month'])['item_cnt_day'].sum().reset_index())
sns.pointplot(x='month', y='item_cnt_day', hue='year', data=grouped)
grouped_price = pd.DataFrame(sales_train.groupby(['year', 'month'])['item_price'].mean().reset_index())
sns.pointplot(x='month', y='item_price', hue='year', data=grouped_price)
ts = sales_train.groupby(['date_block_num'])['item_cnt_day'].sum()
ts.astype('float')
plt.figure(figsize=(16, 8))
plt.title('Total Sales of the whole time period')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.plot(ts)
sns.jointplot(x='item_cnt_day', y='item_price', data=sales_train, height=8)

sales_train.item_cnt_day.hist(bins=100)
sales_train.item_cnt_day.describe()
print('item_price 0ları temizlemeden önce veri kümesi boyutu:', sales_train.shape)
sales_train = sales_train.query('item_price > 0')
print('item_price 0ları temizledikten sonra veri kümesi boyutu:', sales_train.shape)
print('Filtreden önceki veri kümesi boyutu:', sales_train.shape)
sales_train = sales_train[sales_train['shop_id'].isin(test['shop_id'].unique())]
sales_train = sales_train[sales_train['item_id'].isin(test['item_id'].unique())]
print('Filtreden sonraki veri kümesi boyutu:', sales_train.shape)
print('Aykırı değerleri kaldırmadan önce veri kümesi boyutu:', sales_train.shape)
sales_train = sales_train.query('item_cnt_day >= 0 and item_cnt_day <= 125 and item_price < 75000')
print('Aykırı değerleri kaldırdıktan sonra veri kümesi boyutu:', sales_train.shape)
sns.jointplot(x='item_cnt_day', y='item_price', data=sales_train, height=8)

cleaned = pd.DataFrame(sales_train.groupby(['year', 'month'])['item_cnt_day'].sum().reset_index())
sns.pointplot(x='month', y='item_cnt_day', hue='year', data=cleaned)
monthly_sales = sales_train.groupby(['date_block_num', 'shop_id', 'item_id'])['date_block_num', 'date', 'item_price', 'item_cnt_day'].agg({'date_block_num': 'mean', 'date': ['min', 'max'], 'item_price': 'mean', 'item_cnt_day': 'sum'})
monthly_sales.head(5)
sales_data_flat = monthly_sales.item_cnt_day.apply(list).reset_index()
sales_data_flat = pd.merge(test, sales_data_flat, on=['item_id', 'shop_id'], how='left')
sales_data_flat.fillna(0, inplace=True)
sales_data_flat.drop(['shop_id', 'item_id'], inplace=True, axis=1)
sales_data_flat.head(20)
pivoted_sales = sales_data_flat.pivot_table(index='ID', columns='date_block_num', fill_value=0, aggfunc='sum')
pivoted_sales.head(20)
X_train = np.expand_dims(pivoted_sales.values[:, :-1], axis=2)
y_train = pivoted_sales.values[:, -1:]
X_test = np.expand_dims(pivoted_sales.values[:, 1:], axis=2)
print(X_train.shape, y_train.shape, X_test.shape)
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.models import load_model, Model
sales_model = Sequential()
sales_model.add(LSTM(units=64, input_shape=(33, 1)))
sales_model.add(Dropout(0.5))
sales_model.add(Dense(1))
sales_model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])
sales_model.summary()