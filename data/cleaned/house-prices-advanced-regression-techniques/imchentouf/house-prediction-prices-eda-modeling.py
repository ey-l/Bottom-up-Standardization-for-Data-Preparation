import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print('The size of trainsets:    {:,}'.format(df_train.size))
print('The shape of trainsets: ', df_train.shape)
print('The size of testsets:    {:,}'.format(df_test.size))
print('The shape of testsets: ', df_test.shape)
df_train.head()
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum() / df_train.isnull().count() * 100).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total Missed', 'Percent of missing in %'])
missing_data.head(30)
df_test.isnull().sum()
total_test = df_test.isnull().sum().sort_values(ascending=False)
percent_t = (df_test.isnull().sum() / df_test.isnull().count() * 100).sort_values(ascending=False)
missing_data_t = pd.concat([total_test, percent_t], axis=1, keys=['Total Missed', 'Percent of missing in %'])
missing_data_t.head(30)
df_train.SalePrice.describe()
df_train.SalePrice.plot.hist(bins=50, color='skyblue', ec='skyblue')
corr = df_train.corr()[['SalePrice']].abs()
corr.style.background_gradient(cmap='coolwarm')
corr1 = df_train.corr()[['SalePrice']].abs()
corr1 = corr1 > 0.51
corr1.loc[corr1['SalePrice'] == True]
print('The size of corelation data :', corr1.loc[corr1['SalePrice'] == True].shape)
df_test = df_test.fillna(df_test.mean())
columns = ['SalePrice', '1stFlrSF', 'TotalBsmtSF', 'GarageCars', 'GarageArea', 'OverallQual', 'GrLivArea', 'YearBuilt', 'FullBath']
sns.pairplot(df_train[columns], height=1.5)
df_train.YrSold.value_counts()
df_train.YrSold.plot.hist()
ax = sns.boxplot(x='OverallQual', y='SalePrice', data=df_train, palette='Set2')
df_features = ['1stFlrSF', 'TotalBsmtSF', 'GarageCars', 'GarageArea', 'OverallQual', 'GrLivArea', 'YearBuilt', 'FullBath']
for i in df_features:
    df_train.plot.scatter(i, 'SalePrice')
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
import xgboost
Y_train = df_train.SalePrice
X_train = df_train[df_features]
my_model = RandomForestRegressor()