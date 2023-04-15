import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
train_data_path = '_data/input/house-prices-advanced-regression-techniques/train.csv'
test_data_path = '_data/input/house-prices-advanced-regression-techniques/test.csv'
train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)
print('Train dataset have ', train_df.shape[0], ' lines and ', train_df.shape[1], ' columns')
print('Test dataset have ', test_df.shape[0], ' lines and ', test_df.shape[1], ' columns')
train_df.head()
year_seasons_df = train_df[['SalePrice', 'MoSold']].copy()

def setSeason(month):
    if month in (6, 7, 8):
        return 'Summer'
    if month in (11, 10, 9):
        return 'Autumn'
    if month in (12, 1, 2):
        return 'Winter'
    return 'Spring'
year_seasons_df['yearSeason'] = year_seasons_df.MoSold.apply(lambda x: setSeason(x))
year_seasons_df.sort_values(by='SalePrice', inplace=True)
trace = go.Box(x=year_seasons_df.yearSeason, y=year_seasons_df.SalePrice)
data = [trace]
layout = go.Layout(title='Prices x Year Season', yaxis={'title': 'Sale Price'}, xaxis={'title': 'Year Season'})
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
year_seasons_gp_df = year_seasons_df.groupby('yearSeason')['SalePrice'].count().reset_index()
year_seasons_gp_df = pd.DataFrame({'yearSeason': year_seasons_gp_df.yearSeason, 'CountHouse': year_seasons_gp_df.SalePrice})
year_seasons_gp_df.sort_values(by='CountHouse', inplace=True)
trace = go.Bar(x=year_seasons_gp_df.yearSeason, y=year_seasons_gp_df.CountHouse)
data = [trace]
layout = go.Layout(title='Count House x Year Station', yaxis={'title': 'Count House'}, xaxis={'title': 'Year Station'})
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

def labelSeason(x):
    if x == 'Summer':
        return 1
    if x == 'Autumn':
        return 2
    if x == 'Winter':
        return 3
    return 4
year_seasons_df['labelSeason'] = year_seasons_df.yearSeason.apply(lambda x: labelSeason(x))
df_corr_year_seasons = year_seasons_df.corr()
df_corr_year_seasons
year_seasons_sorted_df = year_seasons_df.sort_values(by='MoSold')
year_seasons_sorted_gp_df = year_seasons_df.groupby('MoSold')['SalePrice'].count().reset_index()
df = year_seasons_sorted_gp_df
trace = go.Scatter(x=df.MoSold, y=df.SalePrice, mode='markers+lines', line_shape='spline')
data = [trace]
layout = go.Layout(title="Sales by month's", yaxis={'title': 'Count House'}, xaxis={'title': 'Month sold', 'zeroline': False})
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
lotarea_saleprice_df = train_df[['SalePrice', 'LotArea']]
lotarea_saleprice_df.corr()
train_df = train_df.drop(train_df.loc[train_df['LotArea'] > 70000].index)
train_df = train_df.drop(train_df.loc[train_df['SalePrice'] > 500000].index)
trace = go.Scatter(x=train_df.LotArea, y=train_df.SalePrice, mode='markers')
data = [trace]
layout = go.Layout(title='Lot Area x Sale Price', yaxis={'title': 'Sale Price'}, xaxis={'title': 'Lot Area'})
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
y = np.log(train_df.SalePrice)
X = train_df.copy()
X_test = test_df.copy()
X['AreaUtil'] = X['LotArea'] - (X['MasVnrArea'] + X['GarageArea'] + X['PoolArea'])
X_test['AreaUtil'] = X_test['LotArea'] - (X_test['MasVnrArea'] + X_test['GarageArea'] + X_test['PoolArea'])
X['HavePool'] = X['PoolArea'] > 0
X_test['HavePool'] = X_test['PoolArea'] > 0
X['GarageCars2'] = X['GarageCars'] ** 2
X['GarageCarsSQRT'] = np.sqrt(X['GarageCars'])
X['GarageArea'] = X['GarageArea'] ** 2
X['GarageAreaSQRT'] = np.sqrt(X['GarageArea'])
X['LotArea2'] = X['LotArea'] ** 2
X['LotAreaSQRT'] = np.sqrt(X['LotArea'])
X['AreaUtil2'] = X['AreaUtil'] ** 2
X['AreaUtilSQRT'] = np.sqrt(X['AreaUtil'])
X['GrLivArea2'] = X['GrLivArea'] ** 2
X['GrLivAreaSQRT'] = np.sqrt(X['GrLivArea'])
X_test['GarageCars2'] = X_test['GarageCars'] ** 2
X_test['GarageCarsSQRT'] = np.sqrt(X_test['GarageCars'])
X_test['GarageArea'] = X_test['GarageArea'] ** 2
X_test['GarageAreaSQRT'] = np.sqrt(X_test['GarageArea'])
X_test['LotArea2'] = X_test['LotArea'] ** 2
X_test['LotAreaSQRT'] = np.sqrt(X_test['LotArea'])
X_test['AreaUtil2'] = X_test['AreaUtil'] ** 2
X_test['AreaUtilSQRT'] = np.sqrt(X_test['AreaUtil'])
X_test['GrLivArea2'] = X_test['GrLivArea'] ** 2
X_test['GrLivAreaSQRT'] = np.sqrt(X_test['GrLivArea'])
corrmat = X.corr()
top_corr_features = corrmat.index[abs(corrmat['SalePrice']) > 0]
if 1 == 1:
    plt.figure(figsize=(30, 15))
    g = sns.heatmap(X[top_corr_features].corr(), annot=True, cmap='RdYlGn')
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
X.drop(['SalePrice'], axis=1, inplace=True)
X.drop(['OverallQual'], axis=1, inplace=True)
X.head()
cols_sem_muitos_dados = [col for col in X.columns if X[col].isnull().any()]
for col in cols_sem_muitos_dados:
    X[col].fillna('None')
    X_test[col].fillna('None')
X.head()
print('Train Shape:', X.shape)
print('Test Shape:', X_test.shape)
categorical_cols = [cname for cname in X.columns if X[cname].dtype == 'object']
numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
my_cols = categorical_cols + numerical_cols
X = X[my_cols].copy()
X_test = X_test[my_cols].copy()
X.head()
one_hot_cols = categorical_cols
if 1 == 0:
    x_cat_unique_values = [col for col in X[categorical_cols].columns if len(X[col].unique()) <= 10]
    dict_diff_onehot = set(categorical_cols) - set(x_cat_unique_values)
    one_hot_cols = x_cat_unique_values
    labelEncoder = LabelEncoder()
    for col in list(dict_diff_onehot):
        x_unique = X[col].unique()
        x_test_unique = X_test[col].unique()
        union_uniques = list(x_unique) + list(x_test_unique)
        uniques = list(dict.fromkeys(union_uniques))