import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
tf.__version__
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1.info()
_input1.head(20)
_input1 = _input1.drop('Id', axis=1, inplace=False)
coltodrop = ['Alley', 'PoolQC', 'Fence', 'MiscFeature']
_input1 = _input1.drop(coltodrop, axis=1, inplace=False)
from sklearn.preprocessing import LabelEncoder
le_cols = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']

def convert2num(X_new):
    label_encoder = LabelEncoder()
    for col in le_cols:
        X_new[col] = label_encoder.fit_transform(X_new[col].astype(str))
    return X_new
X = _input1.drop('SalePrice', axis=1)
y = _input1['SalePrice'].to_numpy()
(X.shape, y.shape)
X = convert2num(X)
X[le_cols].head()
X.isna().sum()
X.columns[X.isna().any()].tolist()
X = X.dropna(axis=1, inplace=False)
X.columns[X.isna().any()].tolist()
_input1.columns[_input1.isna().any()].tolist()
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