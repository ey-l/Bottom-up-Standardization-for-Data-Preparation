import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PowerTransformer
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.columns
_input1['SalePrice'].describe()
plt.figure(figsize=(20, 10))
sns.distplot(_input1['SalePrice'])
print('歪度: %f' % _input1['SalePrice'].skew())
print('尖度: %f' % _input1['SalePrice'].kurt())
_input1['TotalSF'] = _input1['1stFlrSF'] + _input1['2ndFlrSF'] + _input1['TotalBsmtSF']
_input0['TotalSF'] = _input0['1stFlrSF'] + _input0['2ndFlrSF'] + _input0['TotalBsmtSF']
plt.figure(figsize=(20, 10))
plt.scatter(_input1['TotalSF'], _input1['SalePrice'])
plt.xlabel('TotalSF')
plt.ylabel('SalePrice')
_input1 = _input1.drop(_input1[(_input1['TotalSF'] > 7500) & (_input1['SalePrice'] < 300000)].index)
plt.figure(figsize=(20, 10))
plt.scatter(_input1['TotalSF'], _input1['SalePrice'])
plt.xlabel('TotalSF')
plt.ylabel('SalePrice')
data = pd.concat([_input1['YearBuilt'], _input1['SalePrice']], axis=1)
plt.figure(figsize=(20, 10))
plt.xticks(rotation='90')
sns.boxplot(x='YearBuilt', y='SalePrice', data=data)
_input1 = _input1.drop(_input1[(_input1['YearBuilt'] < 2000) & (_input1['SalePrice'] > 600000)].index)
data = pd.concat([_input1['YearBuilt'], _input1['SalePrice']], axis=1)
plt.figure(figsize=(20, 10))
plt.xticks(rotation='90')
sns.boxplot(x='YearBuilt', y='SalePrice', data=data)
plt.figure(figsize=(20, 10))
plt.scatter(_input1['OverallQual'], _input1['SalePrice'])
plt.xlabel('OverallQual')
plt.ylabel('SalePrice')
_input1 = _input1.drop(_input1[(_input1['OverallQual'] < 5) & (_input1['SalePrice'] > 200000)].index)
_input1 = _input1.drop(_input1[(_input1['OverallQual'] < 10) & (_input1['SalePrice'] > 500000)].index)
plt.figure(figsize=(20, 10))
plt.scatter(_input1['OverallQual'], _input1['SalePrice'])
plt.xlabel('OverallQual')
plt.ylabel('SalePrice')
train_x = _input1.drop('SalePrice', axis=1)
train_y = _input1['SalePrice']
all_data = pd.concat([train_x, _input0], axis=0, sort=True)
train_ID = _input1['Id']
test_ID = _input0['Id']
all_data = all_data.drop('Id', axis=1, inplace=False)
print('train_x: ' + str(train_x.shape))
print('train_y: ' + str(train_y.shape))
print('test_x: ' + str(_input0.shape))
print('all_data: ' + str(all_data.shape))
all_data_na = all_data.isnull().sum()[all_data.isnull().sum() > 0].sort_values(ascending=False)
all_data_na
plt.figure(figsize=(20, 10))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
na_col_list = all_data.isnull().sum()[all_data.isnull().sum() > 0].index.tolist()
all_data[na_col_list].dtypes.sort_values()
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
float_list = all_data[na_col_list].dtypes[all_data[na_col_list].dtypes == 'float64'].index.tolist()
obj_list = all_data[na_col_list].dtypes[all_data[na_col_list].dtypes == 'object'].index.tolist()
all_data[float_list] = all_data[float_list].fillna(0)
all_data[obj_list] = all_data[obj_list].fillna('None')
all_data.isnull().sum()[all_data.isnull().sum() > 0]
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)
train_y = np.log1p(train_y)
plt.figure(figsize=(20, 10))
sns.distplot(train_y)
num_feats = all_data.dtypes[all_data.dtypes != 'object'].index
skewed_feats = all_data[num_feats].apply(lambda x: x.skew()).sort_values(ascending=False)
plt.figure(figsize=(20, 10))
plt.xticks(rotation='90')
sns.barplot(x=skewed_feats.index, y=skewed_feats)
skewed_feats_over = skewed_feats[abs(skewed_feats) > 0.5].index
for i in skewed_feats_over:
    print(min(all_data[i]))
pt = PowerTransformer()