import pandas as pd
import numpy as np
from sklearn import model_selection
pd.pandas.set_option('display.max_columns', None)
train_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train_data
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(25, 10))
sns.heatmap(train_data.isnull(), cmap='viridis')
Id = train_data['Id']
plt.figure(figsize=(10, 8))
sns.heatmap(train_data.corr())
plt.title('HeatMap- Correlation between predictor Variables')

train_data.corr()['SalePrice'].sort_values()
plt.figure(figsize=(10, 8))
sns.boxplot(x='OverallQual', y='SalePrice', data=train_data)

plt.scatter(x='GrLivArea', y='SalePrice', data=train_data)
plt.ylabel('SalePrice')
plt.xlabel('GrLivArea')

numerical_col = [col for col in train_data.columns if train_data[col].dtypes != 'O']
numerical_col.remove('Id')
year_col = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']
train_data.groupby('YrSold')['SalePrice'].mean().plot()
plt.ylabel('SalePrice')
for i in year_col:
    data1 = train_data.copy()
    if i != 'YrSold':
        data1['new'] = data1['YrSold'] - data1[i]
        sns.scatterplot(x='new', y='SalePrice', data=data1)
        plt.xlabel('Number of years since' + ' ' + i)
        plt.title(i)

discrete_col = [col for col in numerical_col if len(train_data[col].value_counts()) < 20 and col not in year_col]
for i in discrete_col:
    df1 = train_data.copy()
    df1.groupby(i)['SalePrice'].mean().plot.bar()
    plt.ylabel('Sale Price')

train_data.groupby(['YrSold', 'MoSold']).count()['SalePrice'].plot(kind='barh', figsize=(20, 25))
train_data['MiscFeature'] = train_data['MiscFeature'].fillna('None')
train_data['Alley'] = train_data['Alley'].fillna('None')
train_data['Fence'] = train_data['Fence'].fillna('None')
train_data['FireplaceQu'] = train_data['FireplaceQu'].fillna('None')
test_data['MiscFeature'] = test_data['MiscFeature'].fillna('None')
test_data['Alley'] = test_data['Alley'].fillna('None')
test_data['Fence'] = test_data['Fence'].fillna('None')
test_data['FireplaceQu'] = test_data['FireplaceQu'].fillna('None')
train_data['MasVnrArea'] = train_data['MasVnrArea'].fillna(0)
test_data['MasVnrArea'] = test_data['MasVnrArea'].fillna(0)
train_data['MasVnrType'] = train_data['MasVnrType'].fillna('None')
test_data['MasVnrType'] = test_data['MasVnrType'].fillna('None')
train_data['PoolQC'] = train_data['PoolQC'].fillna('None')
test_data['PoolQC'] = test_data['PoolQC'].fillna('None')
Basement_cat = ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2')
for i in Basement_cat:
    train_data[i] = train_data[i].fillna('None')
    test_data[i] = test_data[i].fillna('None')
Basement_num = ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath')
for i in Basement_num:
    train_data[i] = train_data[i].fillna(0)
    test_data[i] = test_data[i].fillna(0)
garage_cat = ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond')
for i in garage_cat:
    train_data[i] = train_data[i].fillna('None')
    test_data[i] = test_data[i].fillna('None')
garage_num = ('GarageYrBlt', 'GarageArea', 'GarageCars')
for i in garage_num:
    train_data[i] = train_data[i].fillna(0)
    test_data[i] = test_data[i].fillna(0)
train_data['LotFrontage'] = train_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
test_data['LotFrontage'] = test_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
numeric_cols = train_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = train_data.select_dtypes(include=['object']).columns.tolist()
year_col = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']
print('Values along with Count in the Categorical Columns', '\n')
for i in categorical_cols:
    print(i)
    print(train_data[i].value_counts(), '\n')
train_data.drop(['Utilities', 'Street', 'PoolQC'], axis=1, inplace=True)
test_data.drop(['Utilities', 'Street', 'PoolQC'], axis=1, inplace=True)
train_data
missing_counts = train_data.isnull().sum().sort_values(ascending=False)
missing_counts[missing_counts > 0]
missing_counts = test_data.isna().sum().sort_values(ascending=False)
missing_counts[missing_counts > 0]
from sklearn.impute import SimpleImputer
numeric_cols.remove('SalePrice')
imputer1 = SimpleImputer(strategy='mean')