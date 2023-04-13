import numpy
import pandas
import sklearn.linear_model
import sklearn.ensemble
import matplotlib
import seaborn
_input0 = pandas.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
_input2 = pandas.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
_input0.head()
_input2.head()
_input0.describe()
_input0.isnull().sum()
_input0['item_price'] = _input0['item_price'].abs()
_input0['item_cnt_day'] = _input0['item_cnt_day'].abs()
_input0 = _input0.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_price': 'last', 'item_cnt_day': 'sum'}).reset_index()
_input0 = _input0.rename(columns={'item_cnt_day': 'item_cnt_month'})
_input0.head()
_input0['item_id'].hist()
correlation = _input0.corr()
matplotlib.pyplot.figure(figsize=(12, 12))
corr_heatmap = seaborn.heatmap(correlation, annot=True, cmap='YlOrRd')
_input2['date_block_num'] = 34
_input2 = _input2[['date_block_num', 'shop_id', 'item_id']]
item_price = dict(_input0.groupby('item_id')['item_price'].last().reset_index().values)
_input2['item_price'] = _input2.item_id.map(item_price)
_input2.head()
_input2['item_price'] = _input2['item_price'].fillna(_input2['item_price'].mean())
_input2['item_price']
x_train = _input0.drop('item_cnt_month', axis=1)
y_train = _input0['item_cnt_month']
x_test = _input2
x_test.head()
linear_model = sklearn.linear_model.LinearRegression()