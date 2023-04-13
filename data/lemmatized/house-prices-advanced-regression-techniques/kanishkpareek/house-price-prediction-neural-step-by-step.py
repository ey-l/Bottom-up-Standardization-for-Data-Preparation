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
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1.head()
_input1.isna().sum()
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
y = _input1['SalePrice'].values
data = pd.concat([_input1, _input0], axis=0, sort=False)
data = data.drop(['SalePrice'], axis=1, inplace=False)
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
missing_values_data = missing_values_data.reset_index(level=0, inplace=False)
missing_values_data.columns = ['Feature', 'Number of Missing Values']
missing_values_data['Percentage of Missing Values'] = 100.0 * missing_values_data['Number of Missing Values'] / len(data)
missing_values_data
data['BsmtFinSF1'] = data['BsmtFinSF1'].fillna(0, inplace=False)
data['BsmtFinSF2'] = data['BsmtFinSF2'].fillna(0, inplace=False)
data['TotalBsmtSF'] = data['TotalBsmtSF'].fillna(0, inplace=False)
data['BsmtUnfSF'] = data['BsmtUnfSF'].fillna(0, inplace=False)
data['Electrical'] = data['Electrical'].fillna('FuseA', inplace=False)
data['KitchenQual'] = data['KitchenQual'].fillna('TA', inplace=False)
data['LotFrontage'] = data['LotFrontage'].fillna(data.groupby('1stFlrSF')['LotFrontage'].transform('mean'), inplace=False)
data['LotFrontage'] = data['LotFrontage'].interpolate(method='linear', inplace=False)
data['MasVnrArea'] = data['MasVnrArea'].fillna(data.groupby('MasVnrType')['MasVnrArea'].transform('mean'), inplace=False)
data['MasVnrArea'] = data['MasVnrArea'].interpolate(method='linear', inplace=False)
for col in NAN_col:
    data_type = data[col].dtype
    if data_type == 'object':
        data[col] = data[col].fillna('NA', inplace=False)
    else:
        data[col] = data[col].fillna(data[col].mean(), inplace=False)
data['Total_Square_Feet'] = data['BsmtFinSF1'] + data['BsmtFinSF2'] + data['1stFlrSF'] + data['2ndFlrSF'] + data['TotalBsmtSF']
data['Total_Bath'] = data['FullBath'] + 0.5 * data['HalfBath'] + data['BsmtFullBath'] + 0.5 * data['BsmtHalfBath']
data['Total_Porch_Area'] = data['OpenPorchSF'] + data['3SsnPorch'] + data['EnclosedPorch'] + data['ScreenPorch'] + data['WoodDeckSF']
data['SqFtPerRoom'] = data['GrLivArea'] / (data['TotRmsAbvGrd'] + data['FullBath'] + data['HalfBath'] + data['KitchenAbvGr'])
data = pd.get_dummies(data)
data.head()
_input1 = data[:1460].copy()
_input0 = data[1460:].copy()
_input1['SalePrice'] = y
_input1.head()
top_features = _input1.corr()[['SalePrice']].sort_values(by=['SalePrice'], ascending=False).head(30)
plt.figure(figsize=(5, 10))
sns.heatmap(top_features, cmap='rainbow', annot=True, annot_kws={'size': 16}, vmin=-1)

def plot_data(col, discrete=False):
    if discrete:
        (fig, ax) = plt.subplots(1, 2, figsize=(14, 6))
        sns.stripplot(x=col, y='SalePrice', data=_input1, ax=ax[0])
        sns.countplot(_input1[col], ax=ax[1])
        fig.suptitle(str(col) + ' Analysis')
    else:
        (fig, ax) = plt.subplots(1, 2, figsize=(12, 6))
        sns.scatterplot(x=col, y='SalePrice', data=_input1, ax=ax[0])
        sns.distplot(_input1[col], kde=False, ax=ax[1])
        fig.suptitle(str(col) + ' Analysis')
plot_data('OverallQual', True)
_input1 = _input1.drop(_input1[(_input1['OverallQual'] == 10) & (_input1['SalePrice'] < 200000)].index)
plot_data('GrLivArea')
plot_data('Total_Bath')
_input1 = _input1.drop(_input1[(_input1['Total_Bath'] > 4) & (_input1['SalePrice'] < 200000)].index)
plot_data('TotalBsmtSF')
_input1 = _input1.drop(_input1[(_input1['TotalBsmtSF'] > 3000) & (_input1['SalePrice'] < 400000)].index)
_input1.reset_index()
clf = IsolationForest(max_samples=100, random_state=42)