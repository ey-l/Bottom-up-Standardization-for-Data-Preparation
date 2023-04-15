import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import norm, skew
import warnings
import math
warnings.filterwarnings('ignore')

train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
train.head()
train['SalePrice'].describe()
k = 1
sns.set_theme()
sns.set_style('darkgrid')
sns.distplot(train['SalePrice'], color='purple')
plt.xticks(rotation=90)
plt.title('Figure {}'.format(k), y=-0.5, fontsize=16)
k = k + 1
(fig, axes) = plt.subplots(1, 2, figsize=(17, 5))
fig.suptitle('Figure {}'.format(k), y=-0.1, fontsize=16)
sns.scatterplot(ax=axes[0], x='GrLivArea', y='SalePrice', color='purple', data=train, alpha=0.5)
sns.boxplot(ax=axes[1], x='OverallQual', y='SalePrice', hue='OverallQual', data=train, color='purple', dodge=False)
axes[1].legend('')
corrmat = train.corr()
(f, ax) = plt.subplots(figsize=(12, 9))
heatm = sns.heatmap(corrmat, vmax=0.9, square=True, cmap='mako_r')
heatm.text(15, 50, 'Figure {}'.format(k), fontsize=16)
k = k + 1
n = 10
cols = corrmat.nlargest(n, 'SalePrice')['SalePrice'].index
cols2 = np.array(list(reversed(cols)))
cm = train[cols2].corr()
mask = np.triu(np.ones_like(cm, dtype=np.bool))
cm2 = cm.iloc[1:, :-1].copy()
(f, ax) = plt.subplots(figsize=(12, 9))
sns.set(font_scale=1.3)
heatmap = sns.heatmap(cm, mask=mask, cbar=False, annot=True, square=True, cmap='mako_r', fmt='.2f', vmin=0.1, vmax=0.85, linewidth=0.3, annot_kws={'size': 10}, yticklabels=cols2, xticklabels=cols2)
k = k + 1
(f, ax) = plt.subplots(figsize=(24, 8))
fig = sns.boxplot(x='YearBuilt', y='SalePrice', hue='YearBuilt', data=train, color='purple', dodge=False)
fig.legend('')
fig.axis(ymin=0, ymax=800000)
plt.xticks(rotation=90)
plt.title('Figure {}'.format(k), y=-0.25, fontsize=16)
k = k + 1
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
pairp = sns.pairplot(train[cols], size=1.5)

total = train.isnull().sum().sort_values(ascending=False)
percentage = (train.isnull().sum() / train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percentage], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
for dataset in (train, test):
    for parameter in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'MSSubClass'):
        dataset[parameter] = dataset[parameter].fillna('None')
    for parameter in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'):
        dataset[parameter] = dataset[parameter].fillna(0)
    for parameter in ('MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Utilities'):
        dataset[parameter] = dataset[parameter].fillna(dataset[parameter].mode()[0])
    dataset['Functional'] = dataset['Functional'].fillna('Typ')
    dataset['LotFrontage'] = dataset.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
train.isnull().sum().max()
test.isnull().sum().max()
train.drop(train[train.GrLivArea > 4500].index, inplace=True)
train.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0, 800000), c='purple', alpha=0.5)
plt.title('Figure {}'.format(k), y=-0.35, fontsize=16)
k = k + 1
for dataset in (train, test):
    dataset['TotalSF'] = dataset['TotalBsmtSF'] + dataset['1stFlrSF'] + dataset['2ndFlrSF']
print('Skewness: %f' % train['SalePrice'].skew())
print('Kurtosis: %f' % train['SalePrice'].kurt())
sns.distplot(train['SalePrice'], color='purple', fit=norm)
plt.xticks(rotation=90)
plt.title('Figure {}'.format(k), y=-0.5, fontsize=16)
k = k + 1
train['SalePrice'] = np.log(train['SalePrice'])
sns.distplot(train['SalePrice'], color='purple', fit=norm)
plt.title('Figure {}'.format(k), y=-0.5, fontsize=16)
print('Skewness: %f' % train['SalePrice'].skew())
print('Kurtosis: %f' % train['SalePrice'].kurt())
num_data = train.dtypes[train.dtypes != 'object'].index
skewed_data = train[num_data].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew': skewed_data})
skewness.head(10)
posSkew = skewness[skewness['Skew'] > 0.75]
print('There are {} skewed numerical features to log-transform'.format(posSkew.shape[0]))
posSkew
skewed_feat = list(posSkew.index)
for dataset in (train, test):
    for parameter in skewed_feat:
        dataset[parameter] = np.log1p(dataset[parameter])
test = test[train.drop('SalePrice', axis=1).columns]
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
train_object = train.select_dtypes('object')
test_object = test.select_dtypes('object')
string_column_names = list(train_object.columns)