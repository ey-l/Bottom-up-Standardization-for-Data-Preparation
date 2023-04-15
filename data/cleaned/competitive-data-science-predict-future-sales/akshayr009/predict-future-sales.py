import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
item_categories_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
shops_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
sales_train_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
items_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
test_df = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
sales_train_df.head()
sales_train_df.shape
sales_train_df.info()
sales_train_df.describe()
sales_train_df.isnull().sum()
sales_train_df['date'] = pd.to_datetime(sales_train_df['date'])
sales_train_df['date']
sales_train_df['month_year'] = sales_train_df['date'].dt.to_period('M')
sales_train_df['month_year']
sales_train_df = sales_train_df[sales_train_df['item_price'] > 0]
sales_train_df = sales_train_df[sales_train_df['item_cnt_day'] > 0]
sales_train_df
monthly_data = sales_train_df.pivot_table(index=['shop_id', 'item_id'], values=['item_cnt_day'], columns=['date_block_num'], fill_value=0, aggfunc='sum')
monthly_data.reset_index(inplace=True)
train_data = monthly_data.drop(columns=['shop_id', 'item_id'], level=0)
train_data.fillna(0, inplace=True)
x_train = np.expand_dims(train_data.values[:, :-1], axis=2)
y_train = train_data.values[:, -1:]
test_rows = monthly_data.merge(test_df, on=['item_id', 'shop_id'], how='right')
x_test = test_rows.drop(test_rows.columns[:5], axis=1).drop('ID', axis=1)
x_test.fillna(0, inplace=True)
x_test = np.expand_dims(x_test, axis=2)
print(x_train.shape, y_train.shape, x_test.shape)
model = tf.keras.models.Sequential()
model.add(LSTM(64, input_shape=(33, 1), return_sequences=False))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])