import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
item_cat = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
sales_train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
item_cat.head()
item_cat.info()
items.head()
items.info()
shops.head()
shops.info()
sales_train.head()
sales_train.info()
sales_train.shape
sales_train.isnull().sum()
sales_train['date'] = pd.to_datetime(sales_train['date'], format='%d.%m.%Y')
sales_train.info()
sales_train.head()
test.head()
test.info()
test.shape
test.isnull().sum()
sales_monthly = sales_train.pivot_table(index=['shop_id', 'item_id'], values=['item_cnt_day'], columns=['date_block_num'], aggfunc='sum').reset_index()
sales_monthly.fillna(0, inplace=True)
sales_monthly.head()
df_combined = pd.merge(test, sales_monthly, how='left', on=['item_id', 'shop_id'])
df_combined.fillna(0, inplace=True)
df_combined.head()
df_combined.drop(['shop_id', 'item_id'], inplace=True, axis=1)
X_train = df_combined.values[:, 1:-1]
y_train = df_combined.values[:, -1:]
X_test = df_combined.values[:, 2:]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
print(X_train.shape, X_test.shape)
model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(33, 1)))
model.add(Dropout(0.2))
'#Adding second LSTM layer\nmodel.add(LSTM(units=50, return_sequences=True)\nmodel.add(Dropout(0.2))\n\n#Adding third LSTM layer\nmodel.add(LSTM(units=50, return_sequences=True)\nmodel.add(Dropout(0.2))\n\n#Adding fourth LSTM layer\nmodel.add(LSTM(units=50, return_sequences=False)\nmodel.add(Dropout(0.2))'
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()