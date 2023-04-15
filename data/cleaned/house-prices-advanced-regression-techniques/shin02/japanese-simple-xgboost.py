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
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_x = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train.columns
train['SalePrice'].describe()
plt.figure(figsize=(20, 10))
sns.distplot(train['SalePrice'])
print('歪度: %f' % train['SalePrice'].skew())
print('尖度: %f' % train['SalePrice'].kurt())
train['TotalSF'] = train['1stFlrSF'] + train['2ndFlrSF'] + train['TotalBsmtSF']
test_x['TotalSF'] = test_x['1stFlrSF'] + test_x['2ndFlrSF'] + test_x['TotalBsmtSF']
plt.figure(figsize=(20, 10))
plt.scatter(train['TotalSF'], train['SalePrice'])
plt.xlabel('TotalSF')
plt.ylabel('SalePrice')
train = train.drop(train[(train['TotalSF'] > 7500) & (train['SalePrice'] < 300000)].index)
plt.figure(figsize=(20, 10))
plt.scatter(train['TotalSF'], train['SalePrice'])
plt.xlabel('TotalSF')
plt.ylabel('SalePrice')
data = pd.concat([train['YearBuilt'], train['SalePrice']], axis=1)
plt.figure(figsize=(20, 10))
plt.xticks(rotation='90')
sns.boxplot(x='YearBuilt', y='SalePrice', data=data)
train = train.drop(train[(train['YearBuilt'] < 2000) & (train['SalePrice'] > 600000)].index)
data = pd.concat([train['YearBuilt'], train['SalePrice']], axis=1)
plt.figure(figsize=(20, 10))
plt.xticks(rotation='90')
sns.boxplot(x='YearBuilt', y='SalePrice', data=data)
plt.figure(figsize=(20, 10))
plt.scatter(train['OverallQual'], train['SalePrice'])
plt.xlabel('OverallQual')
plt.ylabel('SalePrice')
train = train.drop(train[(train['OverallQual'] < 5) & (train['SalePrice'] > 200000)].index)
train = train.drop(train[(train['OverallQual'] < 10) & (train['SalePrice'] > 500000)].index)
plt.figure(figsize=(20, 10))
plt.scatter(train['OverallQual'], train['SalePrice'])
plt.xlabel('OverallQual')
plt.ylabel('SalePrice')
train_x = train.drop('SalePrice', axis=1)
train_y = train['SalePrice']
all_data = pd.concat([train_x, test_x], axis=0, sort=True)
train_ID = train['Id']
test_ID = test_x['Id']
all_data.drop('Id', axis=1, inplace=True)
print('train_x: ' + str(train_x.shape))
print('train_y: ' + str(train_y.shape))
print('test_x: ' + str(test_x.shape))
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