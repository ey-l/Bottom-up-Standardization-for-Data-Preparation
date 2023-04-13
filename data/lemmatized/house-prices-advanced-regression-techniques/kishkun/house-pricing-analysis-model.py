import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
ids = _input0['Id'].values
_input1.describe()
_input1.columns
_input1['SalePrice'].describe()
sns.distplot(_input1['SalePrice'])
y_train = _input1.SalePrice.values
x_train = _input1.drop('SalePrice', 1)
data = pd.concat([_input1['SalePrice'], _input1['OverallQual']], axis=1)
plt.figure(figsize=(8, 6))
sns.boxplot(x='OverallQual', y='SalePrice', data=data)
data = pd.concat([_input1['SalePrice'], _input1['GrLivArea']], axis=1)
data.plot.scatter(x='GrLivArea', y='SalePrice')
data = pd.concat([_input1['SalePrice'], _input1['TotalBsmtSF']], axis=1)
data.plot.scatter(x='TotalBsmtSF', y='SalePrice')
data = pd.concat([_input1['SalePrice'], _input1['Neighborhood']], axis=1)
plt.figure(figsize=(20, 6))
sns.boxplot(x='Neighborhood', y='SalePrice', data=data)
corrmat = _input1.corr()
plt.figure(figsize=(12, 12))
sns.heatmap(corrmat, vmax=0.8, square=True)
total = _input1.isnull().sum().sort_values(ascending=False)
percent = (_input1.isnull().sum() / _input1.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
x_train = x_train.drop(missing_data[missing_data['Total'] > 81].index, 1)
x_train = x_train.apply(lambda x: x.fillna(x.value_counts().index[0]))
x_train.isnull().sum().max()
x_train.shape
_input0.info()
_input0 = _input0.drop(missing_data[missing_data['Total'] > 81].index, 1)
_input0 = _input0.apply(lambda x: x.fillna(x.value_counts().index[0]))
x_train = x_train.drop('Id', axis=1, inplace=False)
_input0 = _input0.drop('Id', axis=1, inplace=False)
x_train.shape
x_train.select_dtypes(include='object').columns
from sklearn.preprocessing import LabelEncoder
cols = x_train.select_dtypes(include='object').columns
for c in cols:
    lbl = LabelEncoder()