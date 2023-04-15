import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

with open('_data/input/house-prices-advanced-regression-techniques/data_description.txt') as file:
    print(file.read())
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
df_train.head()
df_test.head()
df_train.columns
df_train.info()
cat_feats = list(df_train.select_dtypes(include='object').columns)
cat_feats_num = ['MSSubClass', 'OverallQual', 'OverallCond']
cat_feats = cat_feats + cat_feats_num
num_feats = list(set(df_train.columns) - set(cat_feats) - set(['Id', 'SalePrice']))
print('Numerical Features: ', len(num_feats))
for feat in num_feats:
    print(feat, end=', ')
print('Categorical Features: ', len(cat_feats))
for feat in cat_feats:
    print(feat, end=', ')
print(df_train.shape)
print(df_test.shape)
df_train.isnull().sum().sort_values(ascending=False)
df_test.isnull().sum().sort_values(ascending=False)
df_train['PoolQC'].value_counts()
df_train.loc[df_train['PoolArea'] == 0]['PoolQC']
df_train['PoolQC'].fillna('NA', inplace=True)
df_test['PoolQC'].fillna('NA', inplace=True)
df_train.isnull().sum().sort_values(ascending=False)
df_test.isnull().sum().sort_values(ascending=False)
df_train['MiscFeature'].value_counts()
df_train['MiscFeature'].fillna('NA', inplace=True)
df_test['MiscFeature'].fillna('NA', inplace=True)
df_train.isnull().sum().sort_values(ascending=False)
df_test.isnull().sum().sort_values(ascending=False)
df_train['Alley'].value_counts()
df_train['Alley'].fillna('NA', inplace=True)
df_test['Alley'].fillna('NA', inplace=True)
df_train.isnull().sum().sort_values(ascending=False)
df_test.isnull().sum().sort_values(ascending=False)
df_train['Fence'].value_counts()
df_train['Fence'].fillna('NA', inplace=True)
df_test['Fence'].fillna('NA', inplace=True)
df_train.isnull().sum().sort_values(ascending=False)
df_test.isnull().sum().sort_values(ascending=False)
df_train['FireplaceQu'].value_counts()
df_train['FireplaceQu'].fillna('NA', inplace=True)
df_test['FireplaceQu'].fillna('NA', inplace=True)
df_train.isnull().sum().sort_values(ascending=False)
df_test.isnull().sum().sort_values(ascending=False)
df_train[['LotFrontage', 'LotArea', 'LotConfig', 'LotShape']].head()
plt.scatter(df_train['LotArea'], df_train['LotFrontage'])
plt.xlabel('Lot Area')
plt.ylabel('Lot Frontage')
df_train['LotArea'].sort_values(ascending=False)
df_train.iloc[df_train['LotArea'].sort_values(ascending=False).index][['LotArea', 'LotFrontage']]
df_train.loc[df_train['LotArea'] > 70800, 'LotArea'] = 70800
plt.scatter(df_train['LotArea'], df_train['LotFrontage'])
df_train.iloc[df_train['LotFrontage'].sort_values(ascending=False).index][['LotArea', 'LotFrontage']]
df_train.loc[df_train['LotFrontage'] > 185, 'LotFrontage'] = 185
plt.scatter(df_train['LotArea'], df_train['LotFrontage'])
sns.regplot(x='LotArea', y='LotFrontage', data=df_train)
df_train.isnull().sum().sort_values(ascending=False)
from sklearn.linear_model import LinearRegression
lot_notnull = df_train['LotFrontage'].isnull() == False
lot_null = df_train['LotFrontage'].isnull() == True
X_miss = df_train.loc[lot_notnull, ['LotArea', 'LotConfig', 'LotShape']]
X_miss.head()
X_miss = pd.get_dummies(X_miss, prefix=['LC', 'LS'], columns=['LotConfig', 'LotShape'], drop_first=False)
X_miss.head()
X_miss.shape
y_miss = df_train.loc[lot_notnull, 'LotFrontage']
y_miss.shape
X_all = df_train[['LotArea', 'LotConfig', 'LotShape']]
X_all = pd.get_dummies(X_all, prefix=['LC', 'LS'], columns=['LotConfig', 'LotShape'], drop_first=False)
lr_miss = LinearRegression()