import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import seaborn as sns
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input2 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
_input1.info()
_input1.head()
_input1.describe()
_input1 = _input1.drop(columns='Id')
_input0 = _input0.drop(columns='Id')
nan_count = 100 * _input1.isna().sum().sort_values(ascending=False) / _input1.shape[0]
fig = px.bar(x=nan_count.index, y=nan_count.values, labels={'y': 'Nan ammount (%)', 'x': 'Feature'})
fig.show()
_input1 = _input1.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'])
_input0 = _input0.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'])
numeric_features = [feature for feature in _input1.columns if _input1[feature].dtypes != 'object' and feature != 'SalePrice']
categorical_features = [feature for feature in _input1.columns if _input1[feature].dtypes == 'object']
nans = _input1.isna().sum()
nans = nans[nans > 0]
for feature in nans.index:
    _input1[feature] = _input1[feature].fillna(_input1[feature].mode()[0])
nans = _input0.isna().sum()
nans = nans[nans > 0]
for feature in nans.index:
    _input0[feature] = _input0[feature].fillna(_input0[feature].mode()[0])
for feature in categorical_features:
    for (num, value) in enumerate(np.unique(list(_input1[feature].unique()) + list(_input0[feature].unique()))):
        _input1[feature + '_' + str(num)] = pd.Series(_input1[feature] == value, dtype='int')
        _input0[feature + '_' + str(num)] = pd.Series(_input0[feature] == value, dtype='int')
    _input1 = _input1.drop(columns=feature)
    _input0 = _input0.drop(columns=feature)
_input1
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
_input1[numeric_features] = scaler.fit_transform(_input1[numeric_features])
_input0[numeric_features] = scaler.transform(_input0[numeric_features])
x_train = _input1.drop(columns='SalePrice')
y_train = _input1['SalePrice']
pca = PCA(n_components=_input1.shape[1] - 1)
x_train = pca.fit_transform(x_train)
fig = go.Figure()
fig.add_traces(go.Bar(x=np.arange(_input1.shape[1] - 1), y=np.cumsum(pca.explained_variance_ratio_), name='Cumulative Variance'))
n_comp = np.where(np.cumsum(pca.explained_variance_ratio_) > 0.95)[0][0]
fig.add_traces(go.Scatter(x=np.arange(_input1.shape[1] - 1), y=[0.95] * (_input1.shape[1] - 1), name='Variance at 95%'))
fig.update_layout(title='How many components we need?', xaxis_title='Components', yaxis_title='Cumulative Variance', font=dict(family='Arial', size=18))
fig.show()
print('With n_components=' + str(n_comp) + ' we have the 95% of the data variance, so we will choose this value.')
pca = PCA(n_components=n_comp + 50)
x_train = pca.fit_transform(_input1.drop(columns=['SalePrice']))
model = tf.keras.Sequential([layers.Dense(2048, activation='relu'), layers.Dropout(0.5), layers.Dense(2048, activation='relu'), layers.Dropout(0.5), layers.Dense(1)])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adamax(0.001))