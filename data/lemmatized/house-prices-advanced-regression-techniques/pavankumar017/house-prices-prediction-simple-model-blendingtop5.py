import pandas as pd
import numpy as np
from sklearn import model_selection
pd.pandas.set_option('display.max_columns', None)
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(25, 10))
sns.heatmap(_input1.isnull(), cmap='viridis')
Id = _input1['Id']
plt.figure(figsize=(10, 8))
sns.heatmap(_input1.corr())
plt.title('HeatMap- Correlation between predictor Variables')
_input1.corr()['SalePrice'].sort_values()
plt.figure(figsize=(10, 8))
sns.boxplot(x='OverallQual', y='SalePrice', data=_input1)
plt.scatter(x='GrLivArea', y='SalePrice', data=_input1)
plt.ylabel('SalePrice')
plt.xlabel('GrLivArea')
numerical_col = [col for col in _input1.columns if _input1[col].dtypes != 'O']
numerical_col.remove('Id')
year_col = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']
_input1.groupby('YrSold')['SalePrice'].mean().plot()
plt.ylabel('SalePrice')
for i in year_col:
    data1 = _input1.copy()
    if i != 'YrSold':
        data1['new'] = data1['YrSold'] - data1[i]
        sns.scatterplot(x='new', y='SalePrice', data=data1)
        plt.xlabel('Number of years since' + ' ' + i)
        plt.title(i)
discrete_col = [col for col in numerical_col if len(_input1[col].value_counts()) < 20 and col not in year_col]
for i in discrete_col:
    df1 = _input1.copy()
    df1.groupby(i)['SalePrice'].mean().plot.bar()
    plt.ylabel('Sale Price')
_input1.groupby(['YrSold', 'MoSold']).count()['SalePrice'].plot(kind='barh', figsize=(20, 25))
_input1['MiscFeature'] = _input1['MiscFeature'].fillna('None')
_input1['Alley'] = _input1['Alley'].fillna('None')
_input1['Fence'] = _input1['Fence'].fillna('None')
_input1['FireplaceQu'] = _input1['FireplaceQu'].fillna('None')
_input0['MiscFeature'] = _input0['MiscFeature'].fillna('None')
_input0['Alley'] = _input0['Alley'].fillna('None')
_input0['Fence'] = _input0['Fence'].fillna('None')
_input0['FireplaceQu'] = _input0['FireplaceQu'].fillna('None')
_input1['MasVnrArea'] = _input1['MasVnrArea'].fillna(0)
_input0['MasVnrArea'] = _input0['MasVnrArea'].fillna(0)
_input1['MasVnrType'] = _input1['MasVnrType'].fillna('None')
_input0['MasVnrType'] = _input0['MasVnrType'].fillna('None')
_input1['PoolQC'] = _input1['PoolQC'].fillna('None')
_input0['PoolQC'] = _input0['PoolQC'].fillna('None')
Basement_cat = ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2')
for i in Basement_cat:
    _input1[i] = _input1[i].fillna('None')
    _input0[i] = _input0[i].fillna('None')
Basement_num = ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath')
for i in Basement_num:
    _input1[i] = _input1[i].fillna(0)
    _input0[i] = _input0[i].fillna(0)
garage_cat = ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond')
for i in garage_cat:
    _input1[i] = _input1[i].fillna('None')
    _input0[i] = _input0[i].fillna('None')
garage_num = ('GarageYrBlt', 'GarageArea', 'GarageCars')
for i in garage_num:
    _input1[i] = _input1[i].fillna(0)
    _input0[i] = _input0[i].fillna(0)
_input1['LotFrontage'] = _input1.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
_input0['LotFrontage'] = _input0.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
numeric_cols = _input1.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = _input1.select_dtypes(include=['object']).columns.tolist()
year_col = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']
print('Values along with Count in the Categorical Columns', '\n')
for i in categorical_cols:
    print(i)
    print(_input1[i].value_counts(), '\n')
_input1 = _input1.drop(['Utilities', 'Street', 'PoolQC'], axis=1, inplace=False)
_input0 = _input0.drop(['Utilities', 'Street', 'PoolQC'], axis=1, inplace=False)
_input1
missing_counts = _input1.isnull().sum().sort_values(ascending=False)
missing_counts[missing_counts > 0]
missing_counts = _input0.isna().sum().sort_values(ascending=False)
missing_counts[missing_counts > 0]
from sklearn.impute import SimpleImputer
numeric_cols.remove('SalePrice')
imputer1 = SimpleImputer(strategy='mean')