import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from sklearn.model_selection import KFold, GroupKFold
from tensorflow.keras import layers
from sklearn.preprocessing import RobustScaler, StandardScaler
rb = RobustScaler()
sc = StandardScaler()
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
dataset = train.pivot_table(index=['shop_id', 'item_id'], values=['item_cnt_day'], columns=['date_block_num'], fill_value=0, aggfunc='sum')
dataset.reset_index(inplace=True)
dataset = pd.merge(test, dataset, on=['item_id', 'shop_id'], how='left')
dataset.fillna(0, inplace=True)
dataset.drop(['shop_id', 'item_id', 'ID'], inplace=True, axis=1)
dataset.shape
X_train = np.expand_dims(dataset.values[:, :-1], axis=2)
y_train = dataset.values[:, -1:]
X_test = np.expand_dims(dataset.values[:, 1:], axis=2)
print(X_train.shape, y_train.shape, X_test.shape)
save_best = tf.keras.callbacks.ModelCheckpoint('Model.h5', monitor='val_loss', verbose=1, save_best_only=True)

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