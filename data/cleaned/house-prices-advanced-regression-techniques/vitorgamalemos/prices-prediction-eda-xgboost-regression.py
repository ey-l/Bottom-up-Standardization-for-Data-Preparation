import random
import xgboost
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pandas_datareader import data
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.ticker as ticker
from matplotlib.ticker import FixedFormatter, FixedLocator
pd.set_option('display.max_rows', 100)
path_file_csv = '_data/input/house-prices-advanced-regression-techniques/train.csv'
prices_train = pd.read_csv(path_file_csv)
cm = sns.light_palette('green', as_cmap=True)
prices_train.head(30).style.background_gradient(cmap=cm)
pd.DataFrame(prices_train.columns, columns=['name'])
pd.DataFrame(prices_train.dtypes, columns=['type'])
prices_train.loc[:, prices_train.columns != 'Id'].describe().style.background_gradient(cmap=cm)
prices_train.loc[:, prices_train.columns == 'SalePrice'].describe().style.background_gradient(cmap=cm)

def get_random_color():
    r1 = lambda : random.randint(0, 255)
    return '#%02X%02X%02X' % (r1(), r1(), r1())

def get_histplot_central_tendency(df: dict, fields: list):
    for field in fields:
        (f, ax1) = plt.subplots(1, 1, figsize=(15, 5))
        v_dist_1 = df[field].values
        sns.histplot(v_dist_1, ax=ax1, color=get_random_color(), kde=True)
        mean = df[field].mean()
        median = df[field].median()
        mode = df[field].mode().values[0]
        ax1.axvline(mean, color='r', linestyle='--', label='Mean')
        ax1.axvline(median, color='g', linestyle='-', label='Mean')
        ax1.axvline(mode, color='b', linestyle='-', label='Mode')
        ax1.legend()
        plt.title(f'{field} - Histogram analysis')

def get_scatter(df: dict, fields: list):
    ylim = (0, 700000)
    for field in fields:
        df_copy = pd.concat([df['SalePrice'], df[field]], axis=1)
        df_copy.plot.scatter(x=field, y='SalePrice', ylim=ylim, color=get_random_color())
        plt.title(f'{field} - Relationship with SalesPrice')
fields = ['LotArea', 'TotalBsmtSF', 'GrLivArea', 'GarageArea', 'SalePrice']
get_histplot_central_tendency(prices_train, fields)
get_scatter(prices_train, fields[0:4])

def get_headmap_price(df: dict):
    corr = df.corr()
    plt.figure(figsize=(35, 35))
    sns.heatmap(corr, annot=True, cmap='YlGnBu', linewidths=0.1, annot_kws={'fontsize': 10})
    plt.title('Correlation house prices - return rate')
get_headmap_price(prices_train)
pd.DataFrame(prices_train.isnull().sum().sort_values(ascending=False), columns=['count']).style.background_gradient(cmap=cm)

def get_boxplot_price(df: dict, fields: list):
    for field in fields:
        data_copy = pd.concat([df['SalePrice'], df[field]], axis=1)
        (f, ax) = plt.subplots(figsize=(26, 6))
        fig = sns.boxplot(x=field, y='SalePrice', data=data_copy, palette='Set3')
        plt.xticks(rotation=90)
        plt.title(f'Boxplot - {field} x SalePrice')

get_boxplot_price(prices_train, ['YearRemodAdd', 'YearBuilt'])

def get_bar_compare(df: dict, fields: list):
    for field in fields:
        plt.figure(figsize=(15, 6))
        sns.barplot(x=field, y='SalePrice', data=df, palette='Set3')
        plt.xlabel(field)
        plt.ylabel('Sale Price')

get_bar_compare(prices_train, ['MoSold', 'MSSubClass', 'Street', 'MSZoning'])

def cleaning_data_none(prices_train: dict, fields: dict):
    for field in fields:
        prices_train[field].fillna('None', inplace=True)

def cleaning_data_int(prices_train: dict, fields: dict):
    for field in fields:
        prices_train[field].fillna(0, inplace=True)

def cleaning_data_median(prices_train: dict, fields: dict):
    for field in fields:
        prices_train[field].fillna(prices_train[field].median(), inplace=True)
fields_clean_none = ['PoolQC', 'Alley', 'FireplaceQu', 'MasVnrType', 'Electrical', 'BsmtFinType2', 'BsmtFinType1', 'BsmtExposure', 'BsmtQual', 'BsmtCond', 'Fence', 'MiscFeature', 'GarageCond', 'GarageQual', 'GarageFinish', 'GarageType', 'SaleType', 'Utilities', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'Functional']
fields_clean_int = ['GarageYrBlt', 'MSZoning', 'BsmtFinSF1', 'BsmtFullBath', 'BsmtHalfBath']
fields_clean_median = ['LotFrontage', 'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageCars', 'GarageArea']
cleaning_data_none(prices_train, fields_clean_none)
cleaning_data_int(prices_train, fields_clean_int)
cleaning_data_median(prices_train, fields_clean_median)
features = prices_train.columns
features = list(features[1:len(features) - 1])
len(features)
df_types = pd.DataFrame(prices_train.dtypes, columns=['types'])
df_types_object = df_types[df_types['types'] == 'object']
for field_obj in df_types_object.index:
    prices_train[field_obj] = prices_train[field_obj].astype('category').cat.codes
prices_train.head(20).style.background_gradient(cmap=cm)
y = prices_train['SalePrice']
X = prices_train[features]
model = xgboost.XGBRegressor(n_estimators=1000, max_depth=10, eta=0.1, subsample=0.7, colsample_bytree=0.8)
(x_train, x_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)