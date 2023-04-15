import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import tensorflow as tf
from sklearn import preprocessing
import random
tf.random.set_seed(53)
random.seed(53)
BASE = '_data/input/competitive-data-science-predict-future-sales/'
item_cat = pd.read_csv(BASE + 'item_categories.csv')
item = pd.read_csv(BASE + 'items.csv')
sales_train = pd.read_csv(BASE + 'sales_train.csv')
shops = pd.read_csv(BASE + 'shops.csv')
sales_test = pd.read_csv(BASE + 'test.csv')

def basic_eda(df):
    print('---------- TOP 5 RECORDS --------')
    print(df.head(5))
    print('---------- INFO -----------------')
    print(df.info())
    print('---------- Describe -------------')
    print(df.describe())
    print('---------- Columns --------------')
    print(df.columns)
    print('---------- Data Types -----------')
    print(df.dtypes)
    print('------- Missing Values ----------')
    print(df.isnull().sum())
    print('------- NULL values -------------')
    print(df.isna().sum())
    print('----- Shape Of Data -------------')
    print(df.shape)
print('============================= Sales Data =============================')
basic_eda(sales_train)
print('============================= Test data =============================')
basic_eda(sales_test)
print('============================= Item Categories =============================')
basic_eda(item_cat)
print('============================= Items =============================')
basic_eda(item)
print('============================= Shops =============================')
basic_eda(shops)
corr = sales_train.corr()
top_corr_features = corr.index[abs(corr['item_cnt_day']) > 0]
plt.figure(figsize=(6, 6))
g = sns.heatmap(sales_train[top_corr_features].corr(), annot=True, cmap='YlGnBu')
cols = ['item_cnt_day', 'item_price']
(fig, ax) = plt.subplots(ncols=len(cols), figsize=(10 * len(cols), 6), sharex=True)
fig.subplots_adjust(wspace=0.2)
for i in range(len(cols)):
    ax[i].boxplot(sales_train[cols[i]])
    ax[i].set_xlabel(cols[i])
    ax[i].set_ylabel('Count')
outlier1 = sales_train[sales_train['item_cnt_day'] > 2000].index[0]
outlier2 = sales_train[sales_train['item_price'] > 300000].index[0]
sales_train.drop([outlier1, outlier2], axis=0, inplace=True)
sales_train.reset_index(inplace=True, drop=True)
sales_train
cols = ['item_cnt_day', 'item_price']
(fig, ax) = plt.subplots(ncols=len(cols), figsize=(10 * len(cols), 6), sharex=True)
fig.subplots_adjust(wspace=0.2)
for i in range(len(cols)):
    ax[i].plot(sales_train[cols[i]])
    ax[i].set_xlabel(cols[i])
    ax[i].set_ylabel('Count')
dataset = sales_train.pivot_table(index=['shop_id', 'item_id'], values=['item_cnt_day'], columns=['date_block_num'], fill_value=0, aggfunc='sum')
dataset
test_Data = sales_test.copy()
test_Data = test_Data.pivot_table(index=['shop_id', 'item_id'], fill_value=0)
Combine_train_test = pd.merge(test_Data, dataset, how='left', on=['shop_id', 'item_id']).fillna(0)
Combine_train_test = Combine_train_test.sort_values(by='ID')
Combine_train_test.head(10)
sales_train.shape
Combine_train_test.shape
Combine_train_test = Combine_train_test.drop(columns=['ID'])
X_train = np.array(Combine_train_test.values[:, :-1]).reshape(np.array(Combine_train_test.values[:, :-1]).shape[0], np.array(Combine_train_test.values[:, :-1]).shape[1], 1)
y_train = Combine_train_test.values[:, -1:]
X_test = np.array(Combine_train_test.values[:, 1:]).reshape(np.array(Combine_train_test.values[:, 1:]).shape[0], np.array(Combine_train_test.values[:, 1:]).shape[1], 1)
Model_Check_point = tf.keras.callbacks.ModelCheckpoint('Model.h5', monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, verbose=0, mode='auto')
lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=10)
call_backs = [early_stopping_callback, lr_reducer, Model_Check_point]

def build_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True), input_shape=(33, 1)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation='relu', kernel_initializer='uniform'))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002), loss='mse', metrics=['mse'])
    model.summary()
    return model
model = build_model()