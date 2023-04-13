import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print(_input1.info())
print('-*' * 20)
print(_input0.info())
_input2 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
_input2.head()
_input1.columns
_input1.head()
_input1.describe()
_input1['SalePrice'].describe()
import seaborn as sns
from matplotlib import pyplot as plt
sns.distplot(_input1['SalePrice'])
print('Skewness: %f' % _input1['SalePrice'].skew())
print('Kurtosis: %f' % _input1['SalePrice'].kurt())
var = 'GrLivArea'
data = pd.concat([_input1['SalePrice'], _input1[var]], axis=1)
sns.regplot(x=var, y='SalePrice', data=data)
var = 'TotalBsmtSF'
data = pd.concat([_input1['SalePrice'], _input1[var]], axis=1)
sns.regplot(x=var, y='SalePrice', data=data)
var = 'OverallQual'
data = pd.concat([_input1['SalePrice'], _input1[var]], axis=1)
(f, ax) = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y='SalePrice', data=data)
fig.axis(ymin=0, ymax=800000)
var = 'YearBuilt'
data = pd.concat([_input1['SalePrice'], _input1[var]], axis=1)
(f, ax) = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y='SalePrice', data=data)
fig.axis(ymin=0, ymax=800000)
plt.xticks(rotation=90)
corrmat = _input1.corr()
(f, ax) = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)
k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(_input1[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(_input1[cols], size=2.5)
total = _input1.isnull().sum().sort_values(ascending=False)
percent = (_input1.isnull().sum() / _input1.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
_input1 = _input1.drop(missing_data[missing_data['Total'] > 1].index, axis=1)
_input1 = _input1.drop(_input1.loc[_input1['Electrical'].isnull()].index)
_input1.isnull().sum().max()
from sklearn.preprocessing import StandardScaler
saleprice_scaled = StandardScaler().fit_transform(_input1['SalePrice'][:, np.newaxis])
low_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][:10]
high_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)
from scipy.stats import norm
from scipy import stats

def diagnostic_plots(train_df, variable):
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    sns.distplot(_input1[variable], fit=norm)
    plt.subplot(1, 2, 2)
    stats.probplot(_input1[variable], dist='norm', plot=plt)
diagnostic_plots(_input1, 'SalePrice')
_input1['SalePrice'] = np.log(_input1['SalePrice'] + 1)
diagnostic_plots(_input1, 'SalePrice')
diagnostic_plots(_input1, 'GrLivArea')
_input1['GrLivArea'] = np.log(_input1['GrLivArea'] + 1)
diagnostic_plots(_input1, 'GrLivArea')
diagnostic_plots(_input1, 'TotalBsmtSF')
_input1['GrLivArea'] = 1 / (_input1['GrLivArea'] + 1)
diagnostic_plots(_input1, 'GrLivArea')
_input1.head()
_input1.columns
_input1.info()
_input1.describe()
corrmat = _input1.corr()
(f, ax) = plt.subplots(figsize=(50, 20))
sns.heatmap(corrmat, vmax=0.8, square=True, annot=True)
_input1 = _input1.drop(['MSSubClass', 'OverallCond', 'BsmtFinSF2', 'LowQualFinSF', 'BsmtFullBath', 'KitchenAbvGr', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'Id'], axis=1, inplace=False)
_input1.info()
from sklearn.preprocessing import LabelEncoder
cols = ('ExterQual', 'ExterCond', 'HeatingQC', 'KitchenQual', 'Functional', 'LandSlope', 'LotShape', 'PavedDrive', 'Street', 'CentralAir')
for c in cols:
    lbl = LabelEncoder()