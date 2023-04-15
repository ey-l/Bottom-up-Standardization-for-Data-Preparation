import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import norm, skew
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
df_train.columns
print('Number of missing data is', df_train.isna().sum().sum())
missing_values = df_train.isna().sum(axis=0) / df_train.shape[0]
missing_values = missing_values.loc[missing_values > 0]
missing_values.sort_values(ascending=True)
missing_values.plot(kind='bar', title='Missing Columns', ylabel='% of missing', ylim=(0, 1.2), grid=True)
df_train['SalePrice'].describe()
plt.figure(figsize=(20, 12))
sns.distplot(df_train['SalePrice'])
corrmat = df_train.corr()
k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
plt.subplots(figsize=(20, 12))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

from sklearn.ensemble import GradientBoostingRegressor
y_train = df_train['SalePrice']
train_and_test_df = pd.concat([df_train, df_test], axis=0)
train_and_test_df = train_and_test_df.drop(['Id', 'SalePrice'], axis=1)
number_of_missing_df = train_and_test_df.isnull().sum().sort_values()
percent_of_missing_df = (train_and_test_df.isnull().sum() / train_and_test_df.isnull().count() * 100).sort_values()
missing_df = pd.concat([number_of_missing_df, percent_of_missing_df], keys=['total number of missing data', 'total percent of missing data'], axis=1)
print(missing_df.head(10), '\n')
print(missing_df.tail(10))
train_and_test_df = train_and_test_df.drop(missing_df[missing_df['total number of missing data'] > 5].index, axis=1)
train_and_test_df.isnull().sum().sort_values(ascending=False)
numeric_data = [column for column in train_and_test_df.select_dtypes(['int', 'float'])]
categoric_data = [column for column in train_and_test_df.select_dtypes(exclude=['int', 'float'])]
for col in numeric_data:
    train_and_test_df[col].fillna(train_and_test_df[col].median(), inplace=True)
for col in categoric_data:
    train_and_test_df[col].fillna(train_and_test_df[col].value_counts().index[0], inplace=True)
print('Number of missing data is', train_and_test_df.isna().sum().sum())
numeric_data = [column for column in train_and_test_df.select_dtypes(['int', 'float'])]
vars_skewed = train_and_test_df[numeric_data].apply(lambda x: skew(x)).sort_values()
for var in vars_skewed.index:
    train_and_test_df[var] = np.log1p(train_and_test_df[var])
train_and_test_df = train_and_test_df.drop(['GarageArea', '1stFlrSF', 'TotRmsAbvGrd'], axis=1)
train_and_test_df = pd.get_dummies(train_and_test_df, drop_first=True)
train_and_test_df.head()
X_train = train_and_test_df[:len(df_train)]
X_test = train_and_test_df[len(df_train):]
(X_train.shape, X_test.shape)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
reg = GradientBoostingRegressor(random_state=42, loss='ls', learning_rate=0.1)