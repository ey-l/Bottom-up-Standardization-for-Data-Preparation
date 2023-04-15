import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import time
import pylab
import scipy.stats as stats
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
pd.set_option('display.max_columns', 40)
pd.set_option('display.max_rows', 100)
custom_palette = ['#457B9D', '#E63946']
sns.set_palette(sns.color_palette(custom_palette))
import warnings
warnings.filterwarnings(action='ignore')
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col=0)
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col=0)
df_train.head()
df_train.shape
df_test.head()
df_test.shape
df_train.describe()
plt.figure(figsize=(12, 6))
sns.histplot(x='SalePrice', data=df_train, kde=True)
plt.figure(figsize=(12, 6))
stats.probplot(df_train.SalePrice, dist='norm', plot=pylab)
print('Target mean:', round(np.mean(df_train.SalePrice), 1))
print('Target std:', round(np.std(df_train.SalePrice), 1))
print('Skewness: %f' % df_train['SalePrice'].skew())
print('Kurtosis: %f' % df_train['SalePrice'].kurt())
df_train['SalePrice'] = np.log1p(df_train['SalePrice'])
plt.figure(figsize=(12, 6))
sns.histplot(x='SalePrice', data=df_train, kde=True)
plt.figure(figsize=(12, 6))
stats.probplot(df_train.SalePrice, dist='norm', plot=pylab)
print('Target mean:', round(np.mean(df_train.SalePrice), 1))
print('Target std:', round(np.std(df_train.SalePrice), 1))
print('Skewness: %f' % df_train['SalePrice'].skew())
print('Kurtosis: %f' % df_train['SalePrice'].kurt())
df_cat = df_train.select_dtypes('object')
df_num = df_train.select_dtypes(['int64', 'float64'])
for i in df_cat:
    print(i, ':\n', len(df_cat[i].value_counts()), 'modes')
for i in df_num:
    print(i, ':\n', len(df_num[i].value_counts()), 'modes')
for i in df_num:
    if len(df_num[i].value_counts()) < 25:
        df_cat[i] = df_num[i]
        df_num = df_num.drop(i, axis=1)
df_num.columns
df_cat.columns
for i in df_cat.columns:
    plt.figure(figsize=(10, 4))
    sns.countplot(x=df_cat[i], data=df_cat, order=df_cat[i].value_counts().index)
    plt.xticks(rotation=70)
for i in df_num:
    plt.figure(figsize=(4, 6))
    sns.boxplot(y=i, data=df_num)
plt.figure(figsize=(12, 10))
sns.heatmap(np.abs(df_train.corr()), cmap='Reds')
for i in df_num:
    plt.figure(figsize=(12, 6))
    sns.relplot(x=i, y='SalePrice', data=df_num, hue='SalePrice')
for i in df_cat:
    plt.figure(figsize=(12, 6))
    sns.boxenplot(x=i, y='SalePrice', data=df_train)
    plt.xticks(rotation=70)
df_train = df_train[(df_train['LotFrontage'] < 250) & (df_train['LotArea'] < 100000) & (df_train['MasVnrArea'] < 1300) & (df_train['LotFrontage'] < 250) & (df_train['BsmtFinSF1'] < 3000) & (df_train['BsmtFinSF2'] < 1200) & (df_train['TotalBsmtSF'] < 4000) & (df_train['1stFlrSF'] < 4000) & (df_train['GrLivArea'] < 5000) & (df_train['WoodDeckSF'] < 800) & (df_train['OpenPorchSF'] < 450) & (df_train['EnclosedPorch'] < 400) & (df_train['SalePrice'] < np.log(650000))]
df_train.isna().sum()
plt.figure(figsize=(12, 6))
missing_values = round(df_train.isna().sum() * 100 / len(df_train), 2)
missing_values = missing_values[missing_values > 0]
missing_values.sort_values(inplace=True, ascending=False)
sns.barplot(x=missing_values.index, y=missing_values.values)
plt.title('Missing values (%)')
plt.xticks(rotation=70)
df_train = df_train.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'Utilities'], axis=1)
df_test = df_test.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'Utilities'], axis=1)
df_train.isna().sum()
list_quali = ['MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'MSZoning', 'Exterior1st', 'Exterior2nd', 'BsmtFullBath', 'BsmtHalfBath', 'KitchenQual', 'Functional', 'GarageCars', 'SaleType']
list_quanti = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageArea']
list_df = [df_train, df_test]
for i in list_df:
    for j in list_quali:
        i[j] = i[j].fillna(i[j].mode()[0])
    for j in list_quanti:
        i[j] = i[j].fillna(i[j].mean())
df_train.isna().sum()
df_test.isna().sum()
df_train.duplicated().sum()
df_test.duplicated().sum()
df_cat = df_train.select_dtypes('object')
df_num = df_train.select_dtypes(['float64', 'int64'])
df_cat
df_cat.shape
df_num
df_num.shape
le = LabelEncoder()
df_train[df_cat.columns] = df_train[df_cat.columns].apply(le.fit_transform)
df_test[df_cat.columns] = df_test[df_cat.columns].apply(le.fit_transform)
df_train
df_test
df_test[['BsmtFullBath', 'BsmtHalfBath', 'GarageCars']] = df_test[['BsmtFullBath', 'BsmtHalfBath', 'GarageCars']].astype('int64')
df_test
X_cat = list(df_cat.columns)
X_cat.extend(('MSSubClass', 'OverallQual', 'OverallCond', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'MoSold'))
X_cat
df_train = pd.get_dummies(data=df_train, columns=X_cat)
df_train.head()
df_train.shape
df_test = pd.get_dummies(data=df_test, columns=X_cat)
df_test.shape
features = df_train.drop('SalePrice', axis=1)
target = df_train.SalePrice
(X_train, X_test, y_train, y_test) = train_test_split(features, target, test_size=0.15)
skb = SelectKBest(f_regression, k=40)