import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import os
sales_data = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv', dtype={'date': 'str', 'date_block_num': 'int64', 'shop_id': 'int64', 'item_id': 'int64', 'item_price': 'float64', 'item_cnt_day': 'float64'})
item_cat = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv', dtype={'item_category_name': 'str', 'item_category_id': 'int64'})
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv', dtype={'item_name': 'str', 'item_id': 'int64', 'item_category_id': 'int64'})
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv', dtype={'shop_name': 'str', 'shop_id': 'int64'})
test_data = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv', dtype={'ID': 'int64', 'shop_id': 'int64', 'item_id': 'int64'})
print('number of shops: ', shops['shop_id'].max())
print('number of products: ', items['item_id'].max())
print('number of product categories: ', item_cat['item_category_id'].max())
print('number of month: ', sales_data['date_block_num'].max())
print('number of ID: ', test_data['ID'].max())
sales_data.head()
item_cat.head()
items.head()
shops.head()
test_data.head()
sales_data['date'] = pd.to_datetime(sales_data['date'], format='%d.%m.%Y')
dataset = sales_data.pivot_table(index=['shop_id', 'item_id'], values=['item_cnt_day'], columns=['date_block_num'], fill_value=0, aggfunc='sum')
dataset = sales_data.pivot_table(index=['shop_id', 'item_id'], values=['item_cnt_day'], columns=['date_block_num'], fill_value=0, aggfunc='sum')
dataset.reset_index(inplace=True)
dataset.head()
dataset = pd.merge(test_data, dataset, on=['item_id', 'shop_id'], how='left')
dataset.head()
dataset.fillna(0, inplace=True)
dataset.head()
dataset.drop(['shop_id', 'item_id', 'ID'], inplace=True, axis=1)
dataset.head()
X_train = np.expand_dims(dataset.values[:, :-1], axis=2)
y_train = dataset.values[:, -1:]
X_test = np.expand_dims(dataset.values[:, 1:], axis=2)
print(X_train.shape, y_train.shape, X_test.shape)
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation
from keras import backend as K
my_model = Sequential()
my_model.add(LSTM(units=64, input_shape=(33, 1)))
my_model.add(Dense(1))

def softplus(x):
    return np.log(np.exp(x) + 1)
my_model.add(Activation('softplus'))
my_model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])
my_model.summary()