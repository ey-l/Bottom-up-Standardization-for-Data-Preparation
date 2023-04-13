import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1.head()
_input1.shape
_input1.info()
import seaborn as sns
import matplotlib.pyplot as plt
(fig, ax) = plt.subplots(1, 2, figsize=(15, 5))
ax[0].hist(x=_input1.SalePrice)
ax2 = sns.distplot(x=_input1.SalePrice, ax=ax[1])
sns.boxplot(x=_input1.OverallQual, y=_input1.SalePrice)
sns.scatterplot(x=_input1.GrLivArea, y=_input1.SalePrice)
sns.boxplot(x=_input1.TotRmsAbvGrd, y=_input1.SalePrice)
sns.scatterplot(x=_input1.TotalBsmtSF, y=_input1.SalePrice)
cat = ['OverallQual', 'TotRmsAbvGrd', 'GarageCars', 'OverallCond', 'MSSubClass', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'Fireplaces']
y = _input1['SalePrice']
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input0.head()
print('Total no. of train samples: ', len(_input1))
print('Total no. of test samples: ', len(_input0))
total_data = pd.concat([_input1.drop('SalePrice', axis=1), _input0], axis=0, ignore_index=True)
print(total_data.shape)
total_data = total_data.drop('Id', axis=1, inplace=False)
object_type = total_data.dtypes[total_data.dtypes == 'object'].index
object_type
correlation_matrix = _input1.corr()
plt.figure(figsize=(12, 12))
sns.heatmap(correlation_matrix, vmax=0.8, square=True)
k = 10
cols = correlation_matrix.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(_input1[cols].values.T)
sns.set(font_scale=1.25)
plt.figure(figsize=(10, 10))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
null = total_data.isnull().sum()
null_values = pd.DataFrame({'No. of null': null[null != 0].sort_values(ascending=False)})
null_values
total_data = total_data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1, inplace=False)
object_type = object_type.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'])
total_data['FireplaceQu'].describe()
total_data['FireplaceQu'] = total_data['FireplaceQu'].fillna('None')
total_data['LotFrontage'].median()
x = total_data['LotFrontage'].median()
total_data['LotFrontage'] = total_data['LotFrontage'].fillna(x)
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    total_data[col] = total_data[col].fillna('None')
total_data = total_data.drop('GarageYrBlt', axis=1, inplace=False)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    total_data[col] = total_data[col].fillna('None')
total_data['MasVnrArea'] = total_data['MasVnrArea'].fillna(0)
total_data['MasVnrType'].value_counts()
total_data['MasVnrType'] = total_data['MasVnrType'].fillna('None')
total_data['Electrical'].value_counts()
total_data['Electrical'] = total_data['Electrical'].fillna('SBrkr')
total_data['Utilities'].value_counts()
total_data = total_data.drop('Utilities', axis=1, inplace=False)
object_type = object_type.drop('Utilities')
for col in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageArea']:
    total_data[col] = total_data[col].fillna(0)
total_data['MSZoning'] = total_data['MSZoning'].fillna(total_data['MSZoning'].mode()[0])
total_data['MSZoning'].value_counts()
total_data['KitchenQual'] = total_data['KitchenQual'].fillna(total_data['KitchenQual'].mode()[0])
total_data['Exterior1st'] = total_data['Exterior1st'].fillna(total_data['Exterior1st'].mode()[0])
total_data['Exterior2nd'] = total_data['Exterior2nd'].fillna(total_data['Exterior2nd'].mode()[0])
for col in ['GarageCars', 'BsmtFullBath', 'BsmtHalfBath']:
    total_data[col] = total_data[col].fillna(0, inplace=False)
total_data['SaleType'].value_counts()
total_data['SaleType'] = total_data['SaleType'].fillna(total_data['SaleType'].mode()[0])
total_data['Functional'] = total_data['Functional'].fillna('Typ')
total_data.isnull().sum().sum()
total_data['MSSubClass'] = total_data['MSSubClass'].apply(str)
object_type = list(object_type) + ['MSSubClass']
import sklearn
from sklearn.preprocessing import LabelEncoder
for i in object_type:
    le = LabelEncoder()