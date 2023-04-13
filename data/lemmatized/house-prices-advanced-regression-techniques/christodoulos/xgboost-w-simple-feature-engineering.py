from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
full_data = pd.concat([_input1, _input0]).reset_index(drop=True)
sale_price = _input1['SalePrice'].reset_index(drop=True)
del full_data['SalePrice']
print(f'Train dataframe contains {_input1.shape[0]} rows and {_input1.shape[1]} columns.\n')
print(f'Test dataframe contains {_input0.shape[0]} rows and {_input0.shape[1]} columns.\n')
print(f'The merged dataframe contains {full_data.shape[0]} rows and {full_data.shape[1]} columns.')
cols_to_drop = []
for column in full_data:
    if full_data[column].isnull().sum() / len(full_data) >= 0.4:
        cols_to_drop.append(column)
full_data = full_data.drop(cols_to_drop, axis=1, inplace=False)
print(f'{len(cols_to_drop)} columns dropped, the full dataset now comprises of {full_data.shape[1]} variables.')
scaler = MinMaxScaler()
columns = full_data.columns.values
for column in columns:
    if full_data[column].dtype == np.int64 or full_data[column].dtype == np.float64:
        full_data[column] = full_data[column].fillna(full_data[column].median())
        full_data[column] = scaler.fit_transform(np.array(full_data[column]).reshape(-1, 1))
full_data.head()
corr = _input1.corr()
plt.subplots(figsize=(19, 10))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, vmax=0.7, cmap=cmap, square=True)
cols_to_drop = []
sale_price_corr = _input1.corr()['SalePrice'][:-1]
for (column, row) in sale_price_corr.iteritems():
    if abs(float(row)) < 0.12:
        cols_to_drop.append(column)
full_data = full_data.drop(cols_to_drop, axis=1, inplace=False)
print(f'{len(cols_to_drop)} columns dropped, the full dataset now comprises of {full_data.shape[1]} variables.')
count = 0
columns = full_data.columns.values
for column in columns:
    if full_data[column].dtype not in (np.int64, np.float64) and full_data[column].nunique() > 6:
        count += 1
        full_data = full_data.drop(column, axis=1, inplace=False)
print(f'{count} columns dropped, the full dataset now comprises of {full_data.shape[1]} variables.')
full_data = full_data.fillna(full_data.mode().iloc[0])
labelencoder = LabelEncoder()
cols_to_drop = []
columns = full_data.columns.values
for column in columns:
    if full_data[column].dtype not in (np.int64, np.float64) and full_data[column].nunique() > 2:
        dummies = pd.get_dummies(full_data[column], prefix=str(column))
        cols_to_drop.append(column)
        full_data = pd.concat([full_data, dummies], axis=1)
    elif full_data[column].dtype not in (np.int64, np.float64) and full_data[column].nunique() < 3:
        full_data[column] = labelencoder.fit_transform(full_data[column])
        cols_to_drop.append(column)
full_data = full_data.drop(cols_to_drop, axis=1, inplace=False)
print(f'The new dataframe comprises of {_input0.shape[0]} rows and {_input0.shape[1]} columns.\n')
train_df = full_data[:_input1.shape[0]]
test_df = full_data[_input1.shape[0]:]
import tensorflow as tf
tf.random.set_seed(42)
model = tf.keras.Sequential([tf.keras.layers.Dense(1000), tf.keras.layers.BatchNormalization(), tf.keras.layers.Dense(100), tf.keras.layers.Dense(100), tf.keras.layers.Dropout(0.1), tf.keras.layers.Dense(1)])
model.compile(loss=tf.keras.losses.mae, optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003), metrics=['mae'])