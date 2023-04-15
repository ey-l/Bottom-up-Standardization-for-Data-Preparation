import numpy
import pandas
import sklearn.linear_model
import sklearn.ensemble
import matplotlib
import seaborn
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
train_data = pandas.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test_data = pandas.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
train_data.head()
test_data.head()
train_data.describe()
train_data.isnull().sum()
train_data['item_price'] = train_data['item_price'].abs()
train_data['item_cnt_day'] = train_data['item_cnt_day'].abs()
train_data = train_data.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_price': 'last', 'item_cnt_day': 'sum'}).reset_index()
train_data = train_data.rename(columns={'item_cnt_day': 'item_cnt_month'})
train_data.head()
train_data['item_id'].hist()
correlation = train_data.corr()
matplotlib.pyplot.figure(figsize=(12, 12))
corr_heatmap = seaborn.heatmap(correlation, annot=True, cmap='YlOrRd')
test_data['date_block_num'] = 34
test_data = test_data[['date_block_num', 'shop_id', 'item_id']]
item_price = dict(train_data.groupby('item_id')['item_price'].last().reset_index().values)
test_data['item_price'] = test_data.item_id.map(item_price)
test_data.head()
test_data['item_price'] = test_data['item_price'].fillna(test_data['item_price'].mean())
test_data['item_price']
x_train = train_data.drop('item_cnt_month', axis=1)
y_train = train_data['item_cnt_month']
x_test = test_data
x_test.head()
linear_model = sklearn.linear_model.LinearRegression()