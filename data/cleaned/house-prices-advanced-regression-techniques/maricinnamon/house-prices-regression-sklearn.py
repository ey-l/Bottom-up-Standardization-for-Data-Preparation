import math
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, skew
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', header=0)
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
df_train.head(5)
df_test.head(5)
print(df_train.shape)
print(df_test.shape)
fig = plt.figure(figsize=(18, 10))
fig.add_subplot(121)
plt.scatter(x=df_train['GrLivArea'], y=df_train['SalePrice'], color='g', edgecolor='k')
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
fig.add_subplot(122)
plt.scatter(x=df_train['TotalBsmtSF'], y=df_train['SalePrice'], color='m', edgecolor='k')
plt.xlabel('TotalBsmtSF')
plt.ylabel('SalePrice')

stats = df_train['SalePrice'].describe()
stats

def plot_distribution(df):
    fig = plt.figure(figsize=(20, 10))
    df['SalePrice'].plot.kde(color='r')
    df['SalePrice'].plot.hist(density=True, color='blue', edgecolor='k', bins=100)
    plt.legend(['Normal distibution, ($\\mu =${:.2f} and $\\sigma =${:.2f})'.format(stats[1], stats[2])], loc='best')
    plt.title('Frequency distribution plot')
    plt.xlabel('SalePrice')
    plt.ticklabel_format(style='plain', axis='y')
    plt.ticklabel_format(style='plain', axis='x')

plot_distribution(df_train)
df_train['SalePrice'] = np.log1p(df_train['SalePrice'])
plot_distribution(df_train)
cor_matrix = df_train.corr()
cor_matrix.style.background_gradient(cmap='coolwarm')
cor_matrix2 = cor_matrix['SalePrice']
cor_matrix2 = cor_matrix2.to_frame()
cor_matrix2.style.background_gradient(cmap='coolwarm')
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF']
pd.plotting.scatter_matrix(df_train[cols], alpha=0.2, figsize=(25, 25), color='cyan', edgecolor='k')

total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'InPercents'])
missing_data.head(35).style.background_gradient(cmap='autumn')
mask = missing_data['Total'] > 8
missing_data = missing_data.loc[mask]
df_train = df_train.drop(columns=missing_data.index)
df_test = df_test.drop(columns=missing_data.index)
df_train = df_train.fillna('Unknown')
df_test = df_test.fillna('Unknown')
print(df_train.shape)
print(df_test.shape)
for col in df_train:
    df_train[col] = df_train[col].replace('Unknown', df_train[col].mode()[0])
for col in df_test:
    df_test[col] = df_test[col].replace('Unknown', df_test[col].mode()[0])
print(df_train.shape)
print(df_test.shape)
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'InPercents'])
missing_data.head(5).style.background_gradient(cmap='autumn')
df_train.dtypes

def convert_to_categorical(df):
    df['MSSubClass'] = df['MSSubClass'].apply(str)
    df['OverallCond'] = df['OverallCond'].astype(str)
    df['MoSold'] = df['MoSold'].astype(str)
convert_to_categorical(df_train)
convert_to_categorical(df_test)
print(df_train.shape)
print(df_test.shape)
y_train = df_train['SalePrice'].copy()
x_train = df_train.copy().drop(columns=['Id', 'SalePrice'])
x_test = df_test.copy().drop(columns=['Id'])
print(x_train.shape)
print(x_test.shape)
x_train.head()
x_test.head()
x_all = pd.concat([x_train, x_test])
categorical_cols = x_all.select_dtypes(include=np.object).columns
x_all = pd.get_dummies(x_all, prefix=categorical_cols)
x_train = x_all[:len(x_train)]
x_test = x_all[len(x_train):]
print(x_train.shape)
scaler = MinMaxScaler(feature_range=(0, 1))
normed = scaler.fit_transform(x_train.copy())
x_train = pd.DataFrame(data=normed, columns=x_train.columns)
x_train.head()
normed = scaler.fit_transform(x_test.copy())
x_test = pd.DataFrame(data=normed, columns=x_test.columns)
x_test.head()
all_regr_models = [LinearRegression(), Ridge(), RidgeCV(), LassoCV(max_iter=100000), ElasticNetCV()]
all_rmse_train = {}
all_acc_train = {}
for model in all_regr_models:
    model_name = model.__class__.__name__
    print('♦ ', model_name)