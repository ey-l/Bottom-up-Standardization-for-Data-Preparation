import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
ids = df_test['Id'].values
df_train.describe()
df_train.columns
df_train['SalePrice'].describe()
sns.distplot(df_train['SalePrice'])
y_train = df_train.SalePrice.values
x_train = df_train.drop('SalePrice', 1)
data = pd.concat([df_train['SalePrice'], df_train['OverallQual']], axis=1)
plt.figure(figsize=(8, 6))
sns.boxplot(x='OverallQual', y='SalePrice', data=data)
data = pd.concat([df_train['SalePrice'], df_train['GrLivArea']], axis=1)
data.plot.scatter(x='GrLivArea', y='SalePrice')
data = pd.concat([df_train['SalePrice'], df_train['TotalBsmtSF']], axis=1)
data.plot.scatter(x='TotalBsmtSF', y='SalePrice')
data = pd.concat([df_train['SalePrice'], df_train['Neighborhood']], axis=1)
plt.figure(figsize=(20, 6))
sns.boxplot(x='Neighborhood', y='SalePrice', data=data)
corrmat = df_train.corr()
plt.figure(figsize=(12, 12))
sns.heatmap(corrmat, vmax=0.8, square=True)
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
x_train = x_train.drop(missing_data[missing_data['Total'] > 81].index, 1)
x_train = x_train.apply(lambda x: x.fillna(x.value_counts().index[0]))
x_train.isnull().sum().max()
x_train.shape
df_test.info()
df_test = df_test.drop(missing_data[missing_data['Total'] > 81].index, 1)
df_test = df_test.apply(lambda x: x.fillna(x.value_counts().index[0]))
x_train.drop('Id', axis=1, inplace=True)
df_test.drop('Id', axis=1, inplace=True)
x_train.shape
x_train.select_dtypes(include='object').columns
from sklearn.preprocessing import LabelEncoder
cols = x_train.select_dtypes(include='object').columns
for c in cols:
    lbl = LabelEncoder()