import seaborn as sns
sns.set()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_data = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test_data = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
train_data.head()
test_data.head()
train_Data = train_data.copy()
train_Data.isna().sum()
train_Data = train_Data.pivot_table(index=['shop_id', 'item_id'], values=['item_cnt_day'], columns=['date_block_num'], fill_value=0, aggfunc='sum')
train_Data.head()
train_Data.columns
test_Data = test_data.copy()
test_Data = test_Data.pivot_table(index=['shop_id', 'item_id'], fill_value=0)
Combine_train_test = pd.merge(test_Data, train_Data, how='left', on=['shop_id', 'item_id']).fillna(0)
Combine_train_test = Combine_train_test.sort_values(by='ID')
Combine_train_test.head(10)
train_data.shape
Combine_train_test = Combine_train_test.drop(columns=['ID'])
train_data = np.array(Combine_train_test.values[:, :-1]).reshape(np.array(Combine_train_test.values[:, :-1]).shape[0], np.array(Combine_train_test.values[:, :-1]).shape[1], 1)
train_target = Combine_train_test.values[:, -1:]
test_data = np.array(Combine_train_test.values[:, 1:]).reshape(np.array(Combine_train_test.values[:, 1:]).shape[0], np.array(Combine_train_test.values[:, 1:]).shape[1], 1)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, TimeDistributed, Flatten
model = Sequential()
model.add(LSTM(55, return_sequences=True, input_shape=(train_data.shape[1], 1)))
model.add(LSTM(55, return_sequences=True))
model.add(LSTM(55))
model.add(Dense(25))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam', metrics=['mse'])