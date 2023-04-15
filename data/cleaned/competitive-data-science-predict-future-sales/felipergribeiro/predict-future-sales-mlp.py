import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
item_categories = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
items = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
shops = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
sample_submission = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
train['date'] = pd.to_datetime(train['date'], format='%d.%m.%Y')
pt = pd.pivot_table(train, index=['shop_id', 'item_id'], values='item_cnt_day', columns=['date_block_num'], aggfunc=np.sum, fill_value=0)
pt.reset_index(inplace=True)
df = pd.merge(test, pt, on=['shop_id', 'item_id'], how='left')
df.fillna(0, inplace=True)
X_train = df.drop(columns=['shop_id', 'item_id', 'ID', 33], axis=1)
y_train = df[33]
X_test = df.drop(columns=['shop_id', 'item_id', 'ID', 0], axis=1)
X_test.columns = X_train.columns
model = tf.keras.Sequential()
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse', metrics=[tf.keras.metrics.MeanSquaredError(name='mean_squared_error'), tf.keras.metrics.RootMeanSquaredError(name='root_mean_squared_error')])
rlrp = ReduceLROnPlateau(patience=3, verbose=0)
mc = ModelCheckpoint(filepath='bestmodel.h5', verbose=0, monitor='mean_squared_error', mode='min', save_best_only=True)