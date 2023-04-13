import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print('The size of trainsets:    {:,}'.format(_input1.size))
print('The shape of trainsets: ', _input1.shape)
print('The size of testsets:    {:,}'.format(_input0.size))
print('The shape of testsets: ', _input0.shape)
_input1.head()
total = _input1.isnull().sum().sort_values(ascending=False)
percent = (_input1.isnull().sum() / _input1.isnull().count() * 100).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total Missed', 'Percent of missing in %'])
missing_data.head(30)
_input0.isnull().sum()
total_test = _input0.isnull().sum().sort_values(ascending=False)
percent_t = (_input0.isnull().sum() / _input0.isnull().count() * 100).sort_values(ascending=False)
missing_data_t = pd.concat([total_test, percent_t], axis=1, keys=['Total Missed', 'Percent of missing in %'])
missing_data_t.head(30)
_input1.SalePrice.describe()
_input1.SalePrice.plot.hist(bins=50, color='skyblue', ec='skyblue')
corr = _input1.corr()[['SalePrice']].abs()
corr.style.background_gradient(cmap='coolwarm')
corr1 = _input1.corr()[['SalePrice']].abs()
corr1 = corr1 > 0.51
corr1.loc[corr1['SalePrice'] == True]
print('The size of corelation data :', corr1.loc[corr1['SalePrice'] == True].shape)
_input0 = _input0.fillna(_input0.mean())
columns = ['SalePrice', '1stFlrSF', 'TotalBsmtSF', 'GarageCars', 'GarageArea', 'OverallQual', 'GrLivArea', 'YearBuilt', 'FullBath']
sns.pairplot(_input1[columns], height=1.5)
_input1.YrSold.value_counts()
_input1.YrSold.plot.hist()
ax = sns.boxplot(x='OverallQual', y='SalePrice', data=_input1, palette='Set2')
df_features = ['1stFlrSF', 'TotalBsmtSF', 'GarageCars', 'GarageArea', 'OverallQual', 'GrLivArea', 'YearBuilt', 'FullBath']
for i in df_features:
    _input1.plot.scatter(i, 'SalePrice')
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
import xgboost
Y_train = _input1.SalePrice
X_train = _input1[df_features]
my_model = RandomForestRegressor()