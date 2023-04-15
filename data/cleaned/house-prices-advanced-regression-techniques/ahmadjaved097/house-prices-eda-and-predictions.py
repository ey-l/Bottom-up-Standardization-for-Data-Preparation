import warnings
warnings.filterwarnings('ignore')
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train.head(3)
print('Number of rows in training set: ', train.shape[0])
print('Number of columns in training set: ', train.shape[1])
test.head(3)
print('Number of rows in test dataset: ', test.shape[0])
print('Number of columns in test dataset: ', test.shape[1])
df = pd.concat([train.drop('SalePrice', axis=1), test], axis=0)
df.head(3)
print('Number of rows in dataset: ', df.shape[0])
print('Number of columns in dataset: ', df.shape[1])
df.drop('Id', axis=1).describe().T
print('No. of categorical attributes: ', df.select_dtypes(exclude=['int64', 'float64']).columns.size)
print('No. of numerical attributes: ', df.select_dtypes(exclude=['object']).columns.size)
plt.figure(figsize=(20, 6))
sns.heatmap(df.select_dtypes(exclude=['object']).isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Null Values present in Numerical Attributes', fontsize=18)

plt.figure(figsize=(20, 6))
sns.heatmap(df.select_dtypes(exclude=['int64', 'float64']).isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Null Values present in Categorical Attributes', fontsize=18)

null_val = df.isnull().sum() / len(df) * 100
null_val.sort_values(ascending=False, inplace=True)
null_val = pd.DataFrame(null_val, columns=['missing %'])
null_val = null_val[null_val['missing %'] > 0]
sns.set_style('whitegrid')
plt.figure(figsize=(10, 6))
sns.barplot(x=null_val.index, y=null_val['missing %'], palette='Set1')
plt.xticks(rotation=90)

sns.set_style('whitegrid')
df.hist(bins=30, figsize=(20, 15), color='darkgreen')

plt.tight_layout()
plt.figure(figsize=(30, 20))
sns.heatmap(df.corr(), annot=True, cmap='GnBu')
plt.title('Heatmap of all Features', fontsize=18)

sns.set_style('whitegrid')
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols])

plt.figure(figsize=(10, 6))
sns.scatterplot(x='1stFlrSF', y='SalePrice', data=train, color='orange')
plt.title('SalePrice vs. 1stFlrSF')

plt.figure(figsize=(10, 6))
sns.scatterplot(x='GrLivArea', y='SalePrice', data=train, color='limegreen')
plt.title('SalePrice vs. OverallQual')

plt.figure(figsize=(10, 6))
sns.scatterplot(x='TotalBsmtSF', y='SalePrice', data=train, color='royalblue')
plt.title('SalePrice vs. TotalBsmtSF')

plt.figure(figsize=(10, 6))
sns.scatterplot(x='GarageArea', y='SalePrice', data=train, color='royalblue')
plt.title('SalePrice vs. GarageArea')

sns.set_style('whitegrid')
plt.figure(figsize=(10, 6))
sns.boxplot(x='OverallQual', y='SalePrice', data=train, palette='magma')

plt.figure(figsize=(5, 6))
sns.boxplot(x='Street', y='SalePrice', data=train, palette='magma')
plt.title('SalePrice vs. Street')

plt.figure(figsize=(20, 12))
sns.boxplot(x='YearBuilt', y='SalePrice', data=train)
plt.xticks(rotation=90)
plt.title('SalePrice vs. YearBuilt', fontsize=15)

df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    df[col] = df[col].fillna('None')
for col in ['GarageYrBlt', 'GarageArea', 'GarageCars']:
    df[col] = df[col].fillna(int(0))
for col in ('BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtCond', 'BsmtQual'):
    df[col] = df[col].fillna('None')
df['MasVnrArea'] = df['MasVnrArea'].fillna(int(0))
df['MasVnrType'] = df['MasVnrType'].fillna('None')
df['Electrical'] = df['Electrical'].fillna(df['Electrical']).mode()[0]
df = df.drop(['Utilities'], axis=1)
df['PoolQC'] = df['PoolQC'].fillna('None')
df['MiscFeature'].fillna('None', inplace=True)
df['Alley'].fillna('None', inplace=True)
df['Fence'].fillna('None', inplace=True)
df['FireplaceQu'] = df['FireplaceQu'].fillna('None')
df['KitchenQual'].fillna(df['KitchenQual'].mode()[0], inplace=True)
df['BsmtFullBath'].fillna(0, inplace=True)
df['FullBath'].fillna(df['FullBath'].mode()[0], inplace=True)
for col in ['SaleType', 'KitchenQual', 'Exterior2nd', 'Exterior1st', 'Electrical']:
    df[col].fillna(df[col].mode()[0], inplace=True)
df['MSZoning'].fillna(df['MSZoning'].mode()[0], inplace=True)
df['Functional'].fillna(df['Functional'].mode()[0], inplace=True)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    df[col].fillna(0, inplace=True)
plt.figure(figsize=(15, 4))
sns.heatmap(df.isnull(), yticklabels=False)

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope', 'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 'YrSold', 'MoSold', 'MSZoning', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'Foundation', 'GarageType', 'MiscFeature', 'SaleType', 'SaleCondition', 'Electrical', 'Heating')
from sklearn.preprocessing import LabelEncoder
for c in cols:
    lbl = LabelEncoder()