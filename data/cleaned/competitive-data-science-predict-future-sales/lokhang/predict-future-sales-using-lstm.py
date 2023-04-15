import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import timedelta
from keras import backend as K
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.losses import mean_squared_error
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow import keras
np.random.seed(42)
tf.random.set_seed(42)
df_items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
df_item_categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
df_sales_train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv', parse_dates=['date'], dayfirst=True)
df_shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
df_test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
df_items.head(5)
df_items.dtypes
df_item_categories.head(5)
df_item_categories.dtypes
df_shops.head(5)
df_shops.dtypes
df_sales_train.head(5)
df_sales_train.dtypes
df_sales_train['item_cnt_day'].sum()
df_test.head(5)
df_test.dtypes
print(df_items.shape)
print(df_item_categories.shape)
print(df_sales_train.shape)
print(df_shops.shape)
print(df_test.shape)
df_sales_train = df_sales_train[df_sales_train.item_cnt_day < 1000]
df_sales_train.loc[df_sales_train['shop_id'] == 0, ['shop_id']] = 57
df_sales_train.loc[df_sales_train['shop_id'] == 1, ['shop_id']] = 58
df_sales_train.loc[df_sales_train['shop_id'] == 11, ['shop_id']] = 10
df_train = df_sales_train.pivot_table(index=['shop_id', 'item_id'], values=['item_cnt_day'], columns='date_block_num', fill_value=0, aggfunc=np.sum)
df_train = df_train.reset_index()
df_train = df_train.rename(columns={'item_cnt_day': 'item_cnt_mth'})
df_train
df_train.shape
df_test
dataset = df_test.merge(df_train, on=['shop_id', 'item_id'], how='left')
dataset.loc[dataset['ID'].isna(), ['ID']] = '-1'
dataset = dataset.fillna(0)
dataset
X_train = dataset.drop(columns=['shop_id', 'item_id', 'ID']).values[:, :-2]
y_train = dataset.drop(columns=['shop_id', 'item_id', 'ID']).values[:, -2:-1].clip(0, 20)
X_valid = dataset.drop(columns=['shop_id', 'item_id', 'ID']).values[:, 1:-1]
y_valid = dataset.drop(columns=['shop_id', 'item_id', 'ID']).values[:, -1:].clip(0, 20)
X_test = dataset.drop(columns=['shop_id', 'item_id', 'ID']).values[:, 2:]
print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape)
mm_scaler = preprocessing.MinMaxScaler()
X_train = mm_scaler.fit_transform(X_train)
X_valid = mm_scaler.transform(X_valid)
X_test = mm_scaler.transform(X_test)
X_train = X_train[..., np.newaxis]
X_valid = X_valid[..., np.newaxis]
X_test = X_test[..., np.newaxis]
print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape)

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
model = keras.models.Sequential([keras.layers.LSTM(30, return_sequences=True, dropout=0.3, recurrent_dropout=0.3, input_shape=[None, 1]), keras.layers.LSTM(30, dropout=0.3, recurrent_dropout=0.3), keras.layers.Dense(1)])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])
early_stopping = EarlyStopping(patience=5, monitor='val_rmse', mode='min', restore_best_weights=True)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_rmse', factor=0.8, patience=2, mode='auto', cooldown=3, min_lr=1e-05)
model.summary()