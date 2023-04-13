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
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col=0)
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col=0)
_input1.head()
_input1.shape
_input0.head()
_input0.shape
_input1.describe()
plt.figure(figsize=(12, 6))
sns.histplot(x='SalePrice', data=_input1, kde=True)
plt.figure(figsize=(12, 6))
stats.probplot(_input1.SalePrice, dist='norm', plot=pylab)
print('Target mean:', round(np.mean(_input1.SalePrice), 1))
print('Target std:', round(np.std(_input1.SalePrice), 1))
print('Skewness: %f' % _input1['SalePrice'].skew())
print('Kurtosis: %f' % _input1['SalePrice'].kurt())
_input1['SalePrice'] = np.log1p(_input1['SalePrice'])
plt.figure(figsize=(12, 6))
sns.histplot(x='SalePrice', data=_input1, kde=True)
plt.figure(figsize=(12, 6))
stats.probplot(_input1.SalePrice, dist='norm', plot=pylab)
print('Target mean:', round(np.mean(_input1.SalePrice), 1))
print('Target std:', round(np.std(_input1.SalePrice), 1))
print('Skewness: %f' % _input1['SalePrice'].skew())
print('Kurtosis: %f' % _input1['SalePrice'].kurt())
df_cat = _input1.select_dtypes('object')
df_num = _input1.select_dtypes(['int64', 'float64'])
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
sns.heatmap(np.abs(_input1.corr()), cmap='Reds')
for i in df_num:
    plt.figure(figsize=(12, 6))
    sns.relplot(x=i, y='SalePrice', data=df_num, hue='SalePrice')
for i in df_cat:
    plt.figure(figsize=(12, 6))
    sns.boxenplot(x=i, y='SalePrice', data=_input1)
    plt.xticks(rotation=70)
_input1 = _input1[(_input1['LotFrontage'] < 250) & (_input1['LotArea'] < 100000) & (_input1['MasVnrArea'] < 1300) & (_input1['LotFrontage'] < 250) & (_input1['BsmtFinSF1'] < 3000) & (_input1['BsmtFinSF2'] < 1200) & (_input1['TotalBsmtSF'] < 4000) & (_input1['1stFlrSF'] < 4000) & (_input1['GrLivArea'] < 5000) & (_input1['WoodDeckSF'] < 800) & (_input1['OpenPorchSF'] < 450) & (_input1['EnclosedPorch'] < 400) & (_input1['SalePrice'] < np.log(650000))]
_input1.isna().sum()
plt.figure(figsize=(12, 6))
missing_values = round(_input1.isna().sum() * 100 / len(_input1), 2)
missing_values = missing_values[missing_values > 0]
missing_values = missing_values.sort_values(inplace=False, ascending=False)
sns.barplot(x=missing_values.index, y=missing_values.values)
plt.title('Missing values (%)')
plt.xticks(rotation=70)
_input1 = _input1.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'Utilities'], axis=1)
_input0 = _input0.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'Utilities'], axis=1)
_input1.isna().sum()
list_quali = ['MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'MSZoning', 'Exterior1st', 'Exterior2nd', 'BsmtFullBath', 'BsmtHalfBath', 'KitchenQual', 'Functional', 'GarageCars', 'SaleType']
list_quanti = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageArea']
list_df = [_input1, _input0]
for i in list_df:
    for j in list_quali:
        i[j] = i[j].fillna(i[j].mode()[0])
    for j in list_quanti:
        i[j] = i[j].fillna(i[j].mean())
_input1.isna().sum()
_input0.isna().sum()
_input1.duplicated().sum()
_input0.duplicated().sum()
df_cat = _input1.select_dtypes('object')
df_num = _input1.select_dtypes(['float64', 'int64'])
df_cat
df_cat.shape
df_num
df_num.shape
le = LabelEncoder()
_input1[df_cat.columns] = _input1[df_cat.columns].apply(le.fit_transform)
_input0[df_cat.columns] = _input0[df_cat.columns].apply(le.fit_transform)
_input1
_input0
_input0[['BsmtFullBath', 'BsmtHalfBath', 'GarageCars']] = _input0[['BsmtFullBath', 'BsmtHalfBath', 'GarageCars']].astype('int64')
_input0
X_cat = list(df_cat.columns)
X_cat.extend(('MSSubClass', 'OverallQual', 'OverallCond', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'MoSold'))
X_cat
_input1 = pd.get_dummies(data=_input1, columns=X_cat)
_input1.head()
_input1.shape
_input0 = pd.get_dummies(data=_input0, columns=X_cat)
_input0.shape
features = _input1.drop('SalePrice', axis=1)
target = _input1.SalePrice
(X_train, X_test, y_train, y_test) = train_test_split(features, target, test_size=0.15)
skb = SelectKBest(f_regression, k=40)