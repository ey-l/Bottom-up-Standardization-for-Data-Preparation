import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu, linear
from sklearn.metrics import accuracy_score
df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df
df.describe()
df.dtypes.unique()
int_data = df.loc[:, (df.dtypes == 'int64') | (df.dtypes == 'float64')]
object_data = df.loc[:, (df.dtypes == 'object') | (df.columns == 'SalePrice')]
object_data
object_data['Alley'].dtype
column_not_nan = []
for row in object_data.columns:
    if float(0) in object_data[row]:
        column_not_nan.append(row)
    else:
        pass
    print(row, object_data[row].unique())
for i in [2]:
    column_not_nan.pop(i)
column_not_nan
int_data
int_data.isna().any()
int_data['LotFrontage'] = int_data['LotFrontage'].fillna(int_data['LotFrontage'].mean())
int_data['MasVnrArea'] = int_data['MasVnrArea'].fillna(int_data['MasVnrArea'].mean())
int_data['GarageYrBlt'] = int_data['GarageYrBlt'].fillna(int_data['GarageYrBlt'].mean())
for i in int_data.columns:
    int_data.plot(x=i, y='SalePrice', kind='scatter')
xtrain = int_data.loc[:round(len(int_data['SalePrice']) * 0.7), 'MSSubClass':'YrSold']
xcv = int_data.loc[round(len(int_data['SalePrice']) * 0.7):, 'MSSubClass':'YrSold']
xtrain = xtrain.to_numpy()
xcv = xcv.to_numpy()
ytrain = int_data.loc[:round(len(int_data['SalePrice']) * 0.7), 'SalePrice']
ycv = int_data.loc[round(len(int_data['SalePrice']) * 0.7):, 'SalePrice']
ytrain = ytrain.to_numpy()
ycv = ycv.to_numpy()
model = Sequential([tf.keras.layers.Dense(units=351, activation='relu', name='layers1'), tf.keras.layers.Dense(units=107, activation='relu', name='layers2'), tf.keras.layers.Dense(units=35, activation='relu', name='layers3'), tf.keras.layers.Dense(units=4, activation='relu', name='layers4'), tf.keras.layers.Dense(units=1, activation='relu', name='layers5')])
model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))