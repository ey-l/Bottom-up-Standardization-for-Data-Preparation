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
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1.head()
_input1['SalePrice'].describe()
k = 1
sns.set_theme()
sns.set_style('darkgrid')
sns.distplot(_input1['SalePrice'], color='purple')
plt.xticks(rotation=90)
plt.title('Figure {}'.format(k), y=-0.5, fontsize=16)
k = k + 1
(fig, axes) = plt.subplots(1, 2, figsize=(17, 5))
fig.suptitle('Figure {}'.format(k), y=-0.1, fontsize=16)
sns.scatterplot(ax=axes[0], x='GrLivArea', y='SalePrice', color='purple', data=_input1, alpha=0.5)
sns.boxplot(ax=axes[1], x='OverallQual', y='SalePrice', hue='OverallQual', data=_input1, color='purple', dodge=False)
axes[1].legend('')
corrmat = _input1.corr()
(f, ax) = plt.subplots(figsize=(12, 9))
heatm = sns.heatmap(corrmat, vmax=0.9, square=True, cmap='mako_r')
heatm.text(15, 50, 'Figure {}'.format(k), fontsize=16)
k = k + 1
n = 10
cols = corrmat.nlargest(n, 'SalePrice')['SalePrice'].index
cols2 = np.array(list(reversed(cols)))
cm = _input1[cols2].corr()
mask = np.triu(np.ones_like(cm, dtype=np.bool))
cm2 = cm.iloc[1:, :-1].copy()
(f, ax) = plt.subplots(figsize=(12, 9))
sns.set(font_scale=1.3)
heatmap = sns.heatmap(cm, mask=mask, cbar=False, annot=True, square=True, cmap='mako_r', fmt='.2f', vmin=0.1, vmax=0.85, linewidth=0.3, annot_kws={'size': 10}, yticklabels=cols2, xticklabels=cols2)
k = k + 1
(f, ax) = plt.subplots(figsize=(24, 8))
fig = sns.boxplot(x='YearBuilt', y='SalePrice', hue='YearBuilt', data=_input1, color='purple', dodge=False)
fig.legend('')
fig.axis(ymin=0, ymax=800000)
plt.xticks(rotation=90)
plt.title('Figure {}'.format(k), y=-0.25, fontsize=16)
k = k + 1
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
pairp = sns.pairplot(_input1[cols], size=1.5)
total = _input1.isnull().sum().sort_values(ascending=False)
percentage = (_input1.isnull().sum() / _input1.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percentage], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
for dataset in (_input1, _input0):
    for parameter in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'MSSubClass'):
        dataset[parameter] = dataset[parameter].fillna('None')
    for parameter in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'):
        dataset[parameter] = dataset[parameter].fillna(0)
    for parameter in ('MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Utilities'):
        dataset[parameter] = dataset[parameter].fillna(dataset[parameter].mode()[0])
    dataset['Functional'] = dataset['Functional'].fillna('Typ')
    dataset['LotFrontage'] = dataset.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
_input1.isnull().sum().max()
_input0.isnull().sum().max()
_input1 = _input1.drop(_input1[_input1.GrLivArea > 4500].index, inplace=False)
_input1.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0, 800000), c='purple', alpha=0.5)
plt.title('Figure {}'.format(k), y=-0.35, fontsize=16)
k = k + 1
for dataset in (_input1, _input0):
    dataset['TotalSF'] = dataset['TotalBsmtSF'] + dataset['1stFlrSF'] + dataset['2ndFlrSF']
print('Skewness: %f' % _input1['SalePrice'].skew())
print('Kurtosis: %f' % _input1['SalePrice'].kurt())
sns.distplot(_input1['SalePrice'], color='purple', fit=norm)
plt.xticks(rotation=90)
plt.title('Figure {}'.format(k), y=-0.5, fontsize=16)
k = k + 1
_input1['SalePrice'] = np.log(_input1['SalePrice'])
sns.distplot(_input1['SalePrice'], color='purple', fit=norm)
plt.title('Figure {}'.format(k), y=-0.5, fontsize=16)
print('Skewness: %f' % _input1['SalePrice'].skew())
print('Kurtosis: %f' % _input1['SalePrice'].kurt())
num_data = _input1.dtypes[_input1.dtypes != 'object'].index
skewed_data = _input1[num_data].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew': skewed_data})
skewness.head(10)
posSkew = skewness[skewness['Skew'] > 0.75]
print('There are {} skewed numerical features to log-transform'.format(posSkew.shape[0]))
posSkew
skewed_feat = list(posSkew.index)
for dataset in (_input1, _input0):
    for parameter in skewed_feat:
        dataset[parameter] = np.log1p(dataset[parameter])
_input0 = _input0[_input1.drop('SalePrice', axis=1).columns]
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
train_object = _input1.select_dtypes('object')
test_object = _input0.select_dtypes('object')
string_column_names = list(train_object.columns)