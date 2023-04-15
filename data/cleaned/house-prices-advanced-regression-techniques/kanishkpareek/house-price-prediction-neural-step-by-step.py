import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from kerastuner.tuners import RandomSearch
import plotly
import plotly.express as px
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
train.head()
train.isna().sum()
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
y = train['SalePrice'].values
data = pd.concat([train, test], axis=0, sort=False)
data.drop(['SalePrice'], axis=1, inplace=True)
data.head()
data.info()
column_data_type = []
for col in data.columns:
    data_type = data[col].dtype
    if data[col].dtype in ['int64', 'float64']:
        column_data_type.append('numeric')
    else:
        column_data_type.append('categorical')
plt.figure(figsize=(15, 5))
sns.countplot(x=column_data_type)

missing_values = data.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
missing_values
NAN_col = list(missing_values.to_dict().keys())
missing_values_data = pd.DataFrame(missing_values)
missing_values_data.reset_index(level=0, inplace=True)
missing_values_data.columns = ['Feature', 'Number of Missing Values']
missing_values_data['Percentage of Missing Values'] = 100.0 * missing_values_data['Number of Missing Values'] / len(data)
missing_values_data
data['BsmtFinSF1'].fillna(0, inplace=True)
data['BsmtFinSF2'].fillna(0, inplace=True)
data['TotalBsmtSF'].fillna(0, inplace=True)
data['BsmtUnfSF'].fillna(0, inplace=True)
data['Electrical'].fillna('FuseA', inplace=True)
data['KitchenQual'].fillna('TA', inplace=True)
data['LotFrontage'].fillna(data.groupby('1stFlrSF')['LotFrontage'].transform('mean'), inplace=True)
data['LotFrontage'].interpolate(method='linear', inplace=True)
data['MasVnrArea'].fillna(data.groupby('MasVnrType')['MasVnrArea'].transform('mean'), inplace=True)
data['MasVnrArea'].interpolate(method='linear', inplace=True)
for col in NAN_col:
    data_type = data[col].dtype
    if data_type == 'object':
        data[col].fillna('NA', inplace=True)
    else:
        data[col].fillna(data[col].mean(), inplace=True)
data['Total_Square_Feet'] = data['BsmtFinSF1'] + data['BsmtFinSF2'] + data['1stFlrSF'] + data['2ndFlrSF'] + data['TotalBsmtSF']
data['Total_Bath'] = data['FullBath'] + 0.5 * data['HalfBath'] + data['BsmtFullBath'] + 0.5 * data['BsmtHalfBath']
data['Total_Porch_Area'] = data['OpenPorchSF'] + data['3SsnPorch'] + data['EnclosedPorch'] + data['ScreenPorch'] + data['WoodDeckSF']
data['SqFtPerRoom'] = data['GrLivArea'] / (data['TotRmsAbvGrd'] + data['FullBath'] + data['HalfBath'] + data['KitchenAbvGr'])
data = pd.get_dummies(data)
data.head()
train = data[:1460].copy()
test = data[1460:].copy()
train['SalePrice'] = y
train.head()
top_features = train.corr()[['SalePrice']].sort_values(by=['SalePrice'], ascending=False).head(30)
plt.figure(figsize=(5, 10))
sns.heatmap(top_features, cmap='rainbow', annot=True, annot_kws={'size': 16}, vmin=-1)

def plot_data(col, discrete=False):
    if discrete:
        (fig, ax) = plt.subplots(1, 2, figsize=(14, 6))
        sns.stripplot(x=col, y='SalePrice', data=train, ax=ax[0])
        sns.countplot(train[col], ax=ax[1])
        fig.suptitle(str(col) + ' Analysis')
    else:
        (fig, ax) = plt.subplots(1, 2, figsize=(12, 6))
        sns.scatterplot(x=col, y='SalePrice', data=train, ax=ax[0])
        sns.distplot(train[col], kde=False, ax=ax[1])
        fig.suptitle(str(col) + ' Analysis')
plot_data('OverallQual', True)
train = train.drop(train[(train['OverallQual'] == 10) & (train['SalePrice'] < 200000)].index)
plot_data('GrLivArea')
plot_data('Total_Bath')
train = train.drop(train[(train['Total_Bath'] > 4) & (train['SalePrice'] < 200000)].index)
plot_data('TotalBsmtSF')
train = train.drop(train[(train['TotalBsmtSF'] > 3000) & (train['SalePrice'] < 400000)].index)
train.reset_index()
clf = IsolationForest(max_samples=100, random_state=42)