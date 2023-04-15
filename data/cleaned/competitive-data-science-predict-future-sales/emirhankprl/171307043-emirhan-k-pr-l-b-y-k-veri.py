import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import os
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
item_cats = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
sales = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
submission = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
sales['date'] = pd.to_datetime(sales['date'], format='%d.%m.%Y')
dataset = sales.pivot_table(index=['shop_id', 'item_id'], values=['item_cnt_day'], columns=['date_block_num'], fill_value=0, aggfunc='sum')
dataset.reset_index(inplace=True)
dataset = pd.merge(test, dataset, on=['item_id', 'shop_id'], how='left')
dataset.fillna(0, inplace=True)
dataset.drop(['shop_id', 'item_id', 'ID'], inplace=True, axis=1)
X_train = np.expand_dims(dataset.values[:, :-1], axis=2)
y_train = dataset.values[:, -1:]
X_train = X_train.reshape((214200, 33, 1))
y_train = y_train.reshape(214200, 1)
X_test = np.expand_dims(dataset.values[:, 1:], axis=2)
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
model = Sequential()
model.add(LSTM(8, input_shape=(3, 1), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(4, input_shape=(3, 1), return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])