import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
_input0.head()
_input1.info()
_input0.info()
_input1.isnull().sum()
print('The size of train data (row,column) is:' + str(_input1.shape))
print('The size of test data (row, column) is:' + str(_input0.shape))
_input1.columns
_input1.describe().T
_input1['SalePrice'].describe()
sns.histplot(_input1['SalePrice'])
_input1.dtypes.value_counts()
total = _input1.isnull().sum().sort_values(ascending=False)
percent = (_input1.isnull().sum() / _input1.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
_input1 = _input1.drop(missing_data[missing_data['Total'] > 0].index, 1)
_input1.isnull().sum().max()
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
sns.pairplot(_input1[cols], height=2.5)
sns.distplot(_input1['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(_input1['SalePrice'], plot=plt)
_input1['SalePrice'] = np.log(_input1['SalePrice'])
sns.distplot(_input1['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(_input1['SalePrice'], plot=plt)
_input1 = pd.get_dummies(_input1)