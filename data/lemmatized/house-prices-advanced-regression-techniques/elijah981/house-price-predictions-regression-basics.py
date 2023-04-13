import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
with open('_data/input/house-prices-advanced-regression-techniques/data_description.txt') as file:
    print(file.read())
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
_input0.head()
_input1.columns
_input1.info()
cat_feats = list(_input1.select_dtypes(include='object').columns)
cat_feats_num = ['MSSubClass', 'OverallQual', 'OverallCond']
cat_feats = cat_feats + cat_feats_num
num_feats = list(set(_input1.columns) - set(cat_feats) - set(['Id', 'SalePrice']))
print('Numerical Features: ', len(num_feats))
for feat in num_feats:
    print(feat, end=', ')
print('Categorical Features: ', len(cat_feats))
for feat in cat_feats:
    print(feat, end=', ')
print(_input1.shape)
print(_input0.shape)
_input1.isnull().sum().sort_values(ascending=False)
_input0.isnull().sum().sort_values(ascending=False)
_input1['PoolQC'].value_counts()
_input1.loc[_input1['PoolArea'] == 0]['PoolQC']
_input1['PoolQC'] = _input1['PoolQC'].fillna('NA', inplace=False)
_input0['PoolQC'] = _input0['PoolQC'].fillna('NA', inplace=False)
_input1.isnull().sum().sort_values(ascending=False)
_input0.isnull().sum().sort_values(ascending=False)
_input1['MiscFeature'].value_counts()
_input1['MiscFeature'] = _input1['MiscFeature'].fillna('NA', inplace=False)
_input0['MiscFeature'] = _input0['MiscFeature'].fillna('NA', inplace=False)
_input1.isnull().sum().sort_values(ascending=False)
_input0.isnull().sum().sort_values(ascending=False)
_input1['Alley'].value_counts()
_input1['Alley'] = _input1['Alley'].fillna('NA', inplace=False)
_input0['Alley'] = _input0['Alley'].fillna('NA', inplace=False)
_input1.isnull().sum().sort_values(ascending=False)
_input0.isnull().sum().sort_values(ascending=False)
_input1['Fence'].value_counts()
_input1['Fence'] = _input1['Fence'].fillna('NA', inplace=False)
_input0['Fence'] = _input0['Fence'].fillna('NA', inplace=False)
_input1.isnull().sum().sort_values(ascending=False)
_input0.isnull().sum().sort_values(ascending=False)
_input1['FireplaceQu'].value_counts()
_input1['FireplaceQu'] = _input1['FireplaceQu'].fillna('NA', inplace=False)
_input0['FireplaceQu'] = _input0['FireplaceQu'].fillna('NA', inplace=False)
_input1.isnull().sum().sort_values(ascending=False)
_input0.isnull().sum().sort_values(ascending=False)
_input1[['LotFrontage', 'LotArea', 'LotConfig', 'LotShape']].head()
plt.scatter(_input1['LotArea'], _input1['LotFrontage'])
plt.xlabel('Lot Area')
plt.ylabel('Lot Frontage')
_input1['LotArea'].sort_values(ascending=False)
_input1.iloc[_input1['LotArea'].sort_values(ascending=False).index][['LotArea', 'LotFrontage']]
_input1.loc[_input1['LotArea'] > 70800, 'LotArea'] = 70800
plt.scatter(_input1['LotArea'], _input1['LotFrontage'])
_input1.iloc[_input1['LotFrontage'].sort_values(ascending=False).index][['LotArea', 'LotFrontage']]
_input1.loc[_input1['LotFrontage'] > 185, 'LotFrontage'] = 185
plt.scatter(_input1['LotArea'], _input1['LotFrontage'])
sns.regplot(x='LotArea', y='LotFrontage', data=_input1)
_input1.isnull().sum().sort_values(ascending=False)
from sklearn.linear_model import LinearRegression
lot_notnull = _input1['LotFrontage'].isnull() == False
lot_null = _input1['LotFrontage'].isnull() == True
X_miss = _input1.loc[lot_notnull, ['LotArea', 'LotConfig', 'LotShape']]
X_miss.head()
X_miss = pd.get_dummies(X_miss, prefix=['LC', 'LS'], columns=['LotConfig', 'LotShape'], drop_first=False)
X_miss.head()
X_miss.shape
y_miss = _input1.loc[lot_notnull, 'LotFrontage']
y_miss.shape
X_all = _input1[['LotArea', 'LotConfig', 'LotShape']]
X_all = pd.get_dummies(X_all, prefix=['LC', 'LS'], columns=['LotConfig', 'LotShape'], drop_first=False)
lr_miss = LinearRegression()