"""author s_agnik1511"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import gc
import sys
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
'author s_agnik1511'
train_data_path = '_data/input/house-prices-advanced-regression-techniques/train.csv'
test_data_path = '_data/input/house-prices-advanced-regression-techniques/test.csv'
'author s_agnik1511'
train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)
print('Train Shape:', train_df.shape)
print('Test Shape:', test_df.shape)
'author s_agnik1511'
train_df.head()
'author s_agnik1511'
year_stations_df = train_df[['SalePrice', 'MoSold']].copy()

def setStation(month):
    if month in (1, 2, 3):
        return 'Summer'
    if month in (4, 5, 6):
        return 'Autumn'
    if month in (7, 8, 9):
        return 'Winter'
    return 'Spring'
year_stations_df['yearStation'] = year_stations_df.MoSold.apply(lambda x: setStation(x))
year_stations_df.sort_values(by='SalePrice', inplace=True)
trace = go.Box(x=year_stations_df.yearStation, y=year_stations_df.SalePrice)
data = [trace]
layout = go.Layout(title='Prices x Year Station', yaxis={'title': 'Sale Price'}, xaxis={'title': 'Year Station'})
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
'author s_agnik1511'
year_stations_gp_df = year_stations_df.groupby('yearStation')['SalePrice'].count().reset_index()
year_stations_gp_df = pd.DataFrame({'yearStation': year_stations_gp_df.yearStation, 'CountHouse': year_stations_gp_df.SalePrice})
year_stations_gp_df.sort_values(by='CountHouse', inplace=True)
'author s_agnik1511'
trace = go.Bar(x=year_stations_gp_df.yearStation, y=year_stations_gp_df.CountHouse)
data = [trace]
layout = go.Layout(title='Count House x Year Station', yaxis={'title': 'Count House'}, xaxis={'title': 'Year Station'})
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
'author s_agnik1511'

def labelStation(x):
    if x == 'Summer':
        return 1
    if x == 'Autumn':
        return 2
    if x == 'Winter':
        return 3
    return 4
year_stations_df['labelStation'] = year_stations_df.yearStation.apply(lambda x: labelStation(x))
df_corr_year_stations = year_stations_df.corr()
df_corr_year_stations
'author s_agnik1511'
year_stations_sorted_df = year_stations_df.sort_values(by='MoSold')
year_stations_sorted_gp_df = year_stations_df.groupby('MoSold')['SalePrice'].count().reset_index()
'author s_agnik1511'
df = year_stations_sorted_gp_df
trace = go.Scatter(x=df.MoSold, y=df.SalePrice, mode='markers+lines', line_shape='spline')
data = [trace]
layout = go.Layout(title="Prices by month's", yaxis={'title': 'Sale Price'}, xaxis={'title': 'Month sold', 'zeroline': False})
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
trace = go.Scatter(x=train_df.LotArea, y=train_df.SalePrice, mode='markers')
data = [trace]
layout = go.Layout(title='Lot Area x Sale Price', yaxis={'title': 'Sale Price'}, xaxis={'title': 'Lot Area'})
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
trace = go.Box(y=train_df.SalePrice, name='Sale Price')
data = [trace]
layout = go.Layout(title='Distribuiton Sale Price', yaxis={'title': 'Sale Price'})
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
trace = go.Box(y=train_df.LotArea, name='Lot Area')
data = [trace]
layout = go.Layout(title='Distribuiton Lot Area', yaxis={'title': 'Lot Area'})
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
'author s_agnik1511'
lotarea_saleprice_df = train_df[['SalePrice', 'LotArea']]
lotarea_saleprice_df.corr()
'author s_agnik1511'
train_df = train_df.drop(train_df.loc[train_df['LotArea'] > 100000].index)
train_df = train_df.drop(train_df.loc[train_df['SalePrice'] > 500000].index)
'author s_agnik1511'
trace = go.Scatter(x=train_df.LotArea, y=train_df.SalePrice, mode='markers')
data = [trace]
layout = go.Layout(title='Lot Area x Sale Price', yaxis={'title': 'Sale Price'}, xaxis={'title': 'Lot Area'})
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
'author s_agnik1511'

def setStation(month):
    if month in (1, 2, 3):
        return 'Summer'
    if month in (4, 5, 6):
        return 'Autumn'
    if month in (7, 8, 9):
        return 'Winter'
    return 'Spring'
train_df['yearStation'] = train_df.MoSold.apply(lambda x: setStation(x))
test_df['yearStation'] = test_df.MoSold.apply(lambda x: setStation(x))
'author s_agnik1511'
y = np.log(train_df.SalePrice)
X = train_df.copy()
'author s_agnik1511'
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
X.drop(['SalePrice', 'OverallQual'], axis=1, inplace=True)
X_test = test_df.copy()
X.head()
'author s_agnik1511'
X_val = X.isnull().sum() * 100 / len(X)
X_val.loc[X_val > 50.0]
'author s_agnik1511'
colls = [col for col in X.columns if X[col].isnull().sum() * 100 / len(X) > 50.0]
for col in colls:
    X[col].fillna('None')
    X_test[col].fillna('None')
'author s_agnik1511'
print('Train Shape:', X.shape)
print('Test Shape:', X_test.shape)
'author s_agnik1511'
categorical_cols = [cname for cname in X.columns if X[cname].dtype == 'object']
numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
'author s_agnik1511'
my_cols = categorical_cols + numerical_cols
X = X[my_cols].copy()
X_test = X_test[my_cols].copy()
X.head()
'author s_agnik1511'
x_cat_unique_values = [col for col in X[categorical_cols].columns if len(X[col].unique()) <= 5]
dict_diff_onehot = set(categorical_cols) - set(x_cat_unique_values)
one_hot_cols = list(dict_diff_onehot)
'author s_agnik1511'
for col in numerical_cols:
    X['{}_{}'.format(col, 2)] = X[col] ** 2
    X_test['{}_{}'.format(col, 2)] = X_test[col] ** 2
    X['{}_{}'.format(col, 3)] = X[col] ** 3
    X_test['{}_{}'.format(col, 3)] = X_test[col] ** 3
X.head()
'author s_agnik1511'
labelEncoder = LabelEncoder()
for col in x_cat_unique_values:
    x_unique = X[col].unique()
    x_test_unique = X_test[col].unique()
    union_uniques = list(x_unique) + list(x_test_unique)
    uniques = list(dict.fromkeys(union_uniques))