import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('darkgrid')
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.impute import SimpleImputer
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col=0)
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col=0)
print('train: ', train.shape)
print('test: ', test.shape)
train.head()
X = pd.concat([train.drop('SalePrice', axis=1), test], axis=0)
y = train[['SalePrice']]
X.info()
numeric_ = X.select_dtypes(exclude=['object']).drop(['MSSubClass'], axis=1).copy()
numeric_.columns
disc_num_var = ['OverallQual', 'OverallCond', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'MoSold', 'YrSold']
cont_num_var = []
for i in numeric_.columns:
    if i not in disc_num_var:
        cont_num_var.append(i)
cat_train = X.select_dtypes(include=['object']).copy()
cat_train['MSSubClass'] = X['MSSubClass']
cat_train.columns
fig = plt.figure(figsize=(18, 16))
for (index, col) in enumerate(cont_num_var):
    plt.subplot(6, 4, index + 1)
    sns.distplot(numeric_.loc[:, col].dropna(), kde=False)
fig.tight_layout(pad=1.0)
fig = plt.figure(figsize=(14, 15))
for (index, col) in enumerate(cont_num_var):
    plt.subplot(6, 4, index + 1)
    sns.boxplot(y=col, data=numeric_.dropna())
fig.tight_layout(pad=1.0)
fig = plt.figure(figsize=(20, 15))
for (index, col) in enumerate(disc_num_var):
    plt.subplot(5, 3, index + 1)
    sns.countplot(x=col, data=numeric_.dropna())
fig.tight_layout(pad=1.0)
fig = plt.figure(figsize=(18, 20))
for index in range(len(cat_train.columns)):
    plt.subplot(9, 5, index + 1)
    sns.countplot(x=cat_train.iloc[:, index], data=cat_train.dropna())
    plt.xticks(rotation=90)
fig.tight_layout(pad=1.0)
plt.figure(figsize=(14, 12))
correlation = numeric_.corr()
sns.heatmap(correlation, mask=correlation < 0.8, linewidth=0.5, cmap='Blues')
numeric_train = train.select_dtypes(exclude=['object'])
correlation = numeric_train.corr()
correlation[['SalePrice']].sort_values(['SalePrice'], ascending=False)
fig = plt.figure(figsize=(20, 20))
for index in range(len(numeric_train.columns)):
    plt.subplot(10, 5, index + 1)
    sns.scatterplot(x=numeric_train.iloc[:, index], y='SalePrice', data=numeric_train.dropna())
fig.tight_layout(pad=1.0)
X.drop(['GarageYrBlt', 'TotRmsAbvGrd', '1stFlrSF', 'GarageCars'], axis=1, inplace=True)
plt.figure(figsize=(25, 8))
plt.title('Number of missing rows')
missing_count = pd.DataFrame(X.isnull().sum(), columns=['sum']).sort_values(by=['sum'], ascending=False).head(20).reset_index()
missing_count.columns = ['features', 'sum']
sns.barplot(x='features', y='sum', data=missing_count)
X.drop(['PoolQC', 'MiscFeature', 'Alley'], axis=1, inplace=True)
(fig, axes) = plt.subplots(1, 2, figsize=(15, 5))
sns.regplot(x=numeric_train['MoSold'], y='SalePrice', data=numeric_train, ax=axes[0], line_kws={'color': 'black'})
sns.regplot(x=numeric_train['YrSold'], y='SalePrice', data=numeric_train, ax=axes[1], line_kws={'color': 'black'})
fig.tight_layout(pad=2.0)
correlation[['SalePrice']].sort_values(['SalePrice'], ascending=False).tail(10)
X.drop(['MoSold', 'YrSold'], axis=1, inplace=True)
cat_col = X.select_dtypes(include=['object']).columns
overfit_cat = []
for i in cat_col:
    counts = X[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(X) * 100 > 96:
        overfit_cat.append(i)
overfit_cat = list(overfit_cat)
X = X.drop(overfit_cat, axis=1)
num_col = X.select_dtypes(exclude=['object']).drop(['MSSubClass'], axis=1).columns
overfit_num = []
for i in num_col:
    counts = X[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(X) * 100 > 96:
        overfit_num.append(i)
overfit_num = list(overfit_num)
X = X.drop(overfit_num, axis=1)
print('Categorical Features with >96% of the same value: ', overfit_cat)
print('Numerical Features with >96% of the same value: ', overfit_num)
out_col = ['LotFrontage', 'LotArea', 'BsmtFinSF1', 'TotalBsmtSF', 'GrLivArea']
fig = plt.figure(figsize=(20, 5))
for (index, col) in enumerate(out_col):
    plt.subplot(1, 5, index + 1)
    sns.boxplot(y=col, data=X)
fig.tight_layout(pad=1.5)
train = train.drop(train[train['LotFrontage'] > 200].index)
train = train.drop(train[train['LotArea'] > 100000].index)
train = train.drop(train[train['BsmtFinSF1'] > 4000].index)
train = train.drop(train[train['TotalBsmtSF'] > 5000].index)
train = train.drop(train[train['GrLivArea'] > 4000].index)
X.shape
pd.DataFrame(X.isnull().sum(), columns=['sum']).sort_values(by=['sum'], ascending=False).head(15)
cat = ['GarageType', 'GarageFinish', 'BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'GarageCond', 'GarageQual', 'BsmtCond', 'BsmtQual', 'FireplaceQu', 'Fence', 'KitchenQual', 'HeatingQC', 'ExterQual', 'ExterCond']
X[cat] = X[cat].fillna('NA')
cols = ['MasVnrType', 'MSZoning', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Electrical', 'Functional']
X[cols] = X.groupby('Neighborhood')[cols].transform(lambda x: x.fillna(x.mode()[0]))
print('Mean of LotFrontage: ', X['LotFrontage'].mean())
print('Mean of GarageArea: ', X['GarageArea'].mean())
neigh_lot = X.groupby('Neighborhood')['LotFrontage'].mean().reset_index(name='LotFrontage_mean')
neigh_garage = X.groupby('Neighborhood')['GarageArea'].mean().reset_index(name='GarageArea_mean')
(fig, axes) = plt.subplots(1, 2, figsize=(22, 8))
axes[0].tick_params(axis='x', rotation=90)
sns.barplot(x='Neighborhood', y='LotFrontage_mean', data=neigh_lot, ax=axes[0])
axes[1].tick_params(axis='x', rotation=90)
sns.barplot(x='Neighborhood', y='GarageArea_mean', data=neigh_garage, ax=axes[1])
X['LotFrontage'] = X.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.mean()))
X['GarageArea'] = X.groupby('Neighborhood')['GarageArea'].transform(lambda x: x.fillna(x.mean()))
X['MSZoning'] = X.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
cont = ['BsmtHalfBath', 'BsmtFullBath', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'MasVnrArea']
X[cont] = X[cont] = X[cont].fillna(X[cont].mean())
X['MSSubClass'] = X['MSSubClass'].apply(str)
ordinal_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
fintype_map = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0}
expose_map = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0}
fence_map = {'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'NA': 0}
ord_col = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'GarageQual', 'GarageCond', 'FireplaceQu']
for col in ord_col:
    X[col] = X[col].map(ordinal_map)
fin_col = ['BsmtFinType1', 'BsmtFinType2']
for col in fin_col:
    X[col] = X[col].map(fintype_map)
X['BsmtExposure'] = X['BsmtExposure'].map(expose_map)
X['Fence'] = X['Fence'].map(fence_map)
X['TotalLot'] = X['LotFrontage'] + X['LotArea']
X['TotalBsmtFin'] = X['BsmtFinSF1'] + X['BsmtFinSF2']
X['TotalSF'] = X['TotalBsmtSF'] + X['2ndFlrSF']
X['TotalBath'] = X['FullBath'] + X['HalfBath']
X['TotalPorch'] = X['OpenPorchSF'] + X['EnclosedPorch'] + X['ScreenPorch']
colum = ['MasVnrArea', 'TotalBsmtFin', 'TotalBsmtSF', '2ndFlrSF', 'WoodDeckSF', 'TotalPorch']
for col in colum:
    col_name = col + '_bin'
    X[col_name] = X[col].apply(lambda x: 1 if x > 0 else 0)
X = pd.get_dummies(X)
plt.figure(figsize=(10, 6))
plt.title('Before transformation of SalePrice')
dist = sns.distplot(train['SalePrice'], norm_hist=False)
plt.figure(figsize=(10, 6))
plt.title('After transformation of SalePrice')
dist = sns.distplot(np.log(train['SalePrice']), norm_hist=False)
y['SalePrice'] = np.log(y['SalePrice'])
x = X.loc[train.index]
y = y.loc[train.index]
test = X.loc[test.index]
from sklearn.preprocessing import RobustScaler
cols = x.select_dtypes(np.number).columns