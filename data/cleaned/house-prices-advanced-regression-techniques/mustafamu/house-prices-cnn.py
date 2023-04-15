import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import tensorflow as tf
tf.__version__
data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
data.info()
data.head(20)
data.drop('Id', axis=1, inplace=True)
coltodrop = ['Alley', 'PoolQC', 'Fence', 'MiscFeature']
data.drop(coltodrop, axis=1, inplace=True)
from sklearn.preprocessing import LabelEncoder
le_cols = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']

def convert2num(X_new):
    label_encoder = LabelEncoder()
    for col in le_cols:
        X_new[col] = label_encoder.fit_transform(X_new[col].astype(str))
    return X_new
X = data.drop('SalePrice', axis=1)
y = data['SalePrice'].to_numpy()
(X.shape, y.shape)
X = convert2num(X)
X[le_cols].head()
X.isna().sum()
X.columns[X.isna().any()].tolist()
X.dropna(axis=1, inplace=True)
X.columns[X.isna().any()].tolist()
data.columns[data.isna().any()].tolist()
col_names = X.columns
col_names
pd.DataFrame(X).isna().sum()
len(X)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
from sklearn.model_selection import train_test_split
tf.random.set_seed(42)
(X_train_vaild, X_test, y_train_vaild, y_test) = train_test_split(X, y, test_size=0.05, random_state=42)
(X_train_vaild.shape, X_test.shape, y_train_vaild.shape, y_test.shape)
tf.random.set_seed(42)
(X_train, X_valid, y_train, y_valid) = train_test_split(X_train_vaild, y_train_vaild, test_size=0.25, random_state=42)
(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)
tf.random.set_seed(42)
model_1 = tf.keras.Sequential([tf.keras.layers.Dense(250, activation='relu'), tf.keras.layers.Dense(10, activation='relu'), tf.keras.layers.Dense(100, activation='relu'), tf.keras.layers.Dense(1)])
model_1.compile(loss=tf.keras.losses.mae, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['MAE'])