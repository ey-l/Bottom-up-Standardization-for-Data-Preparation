import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
import os
for (dirname, _, filenames) in os.walk('_data/input/competitive-data-science-predict-future-sales'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
item_categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
sale_item = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
shop_name = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
train_data = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test_data = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
train_data.head()
test_data.head()
train_Data = train_data.copy()
train_Data.isna().sum()

def month_column(col):
    temp = col.split('.')[1]
    return temp
train_Data['Month'] = train_Data['date'].apply(month_column)

def year_column(col):
    temp = col.split('.')[2]
    return temp
train_Data['Year'] = train_Data['date'].apply(year_column)
train_Data['Sales'] = train_Data['item_price'] * train_Data['item_cnt_day']
item_categories = []
for i in train_Data['item_id']:
    item_categories.append(sale_item['item_category_id'].iloc[i])
train_Data['item_categories'] = item_categories
train_Data['item_id_categories'] = train_Data['item_id'].apply(str) + ',' + train_Data['item_categories'].apply(str)
train_Data.head()
train_Data = train_Data[train_Data['Month'] == '11']
training_data = train_Data.drop(columns=['date', 'date_block_num', 'item_price', 'Month', 'Year', 'Sales', 'item_id_categories', 'item_cnt_day'])
training_target = train_Data['item_cnt_day']
training_data = np.array(training_data)
training_target = np.array(training_target)
training_data.shape
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
model = Sequential()
model.add(Dense(4, activation='sigmoid', input_dim=training_data.shape[1]))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])