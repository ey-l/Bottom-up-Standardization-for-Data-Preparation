import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import cufflinks as cf
cf.go_offline()
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
_input1.shape
_input0.head()
_input0.shape
plt.figure(figsize=(12, 8))
sns.heatmap(_input1.isnull(), yticklabels=False, cbar=False)
train_nan_vals = dict(_input1.isnull().sum())
for (i, j) in train_nan_vals.items():
    print(i, '-->', j)
_input1['LotFrontage'] = _input1['LotFrontage'].fillna(_input1['LotFrontage'].mean())
_input1 = _input1.drop(['Alley'], axis=1, inplace=False)
_input1['MasVnrType'] = _input1['MasVnrType'].fillna(_input1['MasVnrType'].mode()[0])
_input1['MasVnrArea'] = _input1['MasVnrArea'].fillna(_input1['MasVnrArea'].mode()[0])
_input1['BsmtQual'] = _input1['BsmtQual'].fillna(_input1['BsmtQual'].mode()[0])
_input1['BsmtCond'] = _input1['BsmtCond'].fillna(_input1['BsmtCond'].mode()[0])
_input1['BsmtExposure'] = _input1['BsmtExposure'].fillna(_input1['BsmtExposure'].mode()[0])
_input1['BsmtFinType1'] = _input1['BsmtFinType1'].fillna(_input1['BsmtFinType1'].mode()[0])
_input1['BsmtFinType2'] = _input1['BsmtFinType2'].fillna(_input1['BsmtFinType2'].mode()[0])
_input1['Electrical'] = _input1['Electrical'].fillna(_input1['Electrical'].mode()[0])
_input1['FireplaceQu'] = _input1['FireplaceQu'].fillna(_input1['FireplaceQu'].mode()[0])
_input1['GarageType'] = _input1['GarageType'].fillna(_input1['GarageType'].mode()[0])
_input1['GarageYrBlt'] = _input1['GarageYrBlt'].fillna(_input1['GarageYrBlt'].mean())
_input1['GarageFinish'] = _input1['GarageFinish'].fillna(_input1['GarageFinish'].mode()[0])
_input1['GarageQual'] = _input1['GarageQual'].fillna(_input1['GarageQual'].mode()[0])
_input1['GarageCond'] = _input1['GarageCond'].fillna(_input1['GarageCond'].mode()[0])
_input1 = _input1.drop(['Id', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=False)
_input1.isnull().sum().any()
_input1.shape
_input1.isnull().sum().any()
_input0.info()
plt.figure(figsize=(12, 6))
sns.heatmap(_input0.isnull(), cmap='viridis')
test_nan_vals = dict(_input0.isnull().sum())
for (i, j) in test_nan_vals.items():
    print(i, '--->', j)
_input0['BsmtFinType2'].value_counts()
_input0['MSZoning'] = _input0['MSZoning'].fillna(_input0['MSZoning'].mode()[0])
_input0['LotFrontage'] = _input0['LotFrontage'].fillna(_input0['LotFrontage'].mean())
_input0['Utilities'] = _input0['Utilities'].fillna(_input0['Utilities'].mode()[0])
_input0['Exterior1st'] = _input0['Exterior1st'].fillna(_input0['Exterior1st'].mode()[0])
_input0['Exterior2nd'] = _input0['Exterior2nd'].fillna(_input0['Exterior2nd'].mode()[0])
_input0['MasVnrType'] = _input0['MasVnrType'].fillna(_input0['MasVnrType'].mode()[0])
_input0['MasVnrArea'] = _input0['MasVnrArea'].fillna(_input0['MasVnrArea'].mean())
_input0['BsmtQual'] = _input0['BsmtQual'].fillna(_input0['BsmtQual'].mode()[0])
_input0['MasVnrType'] = _input0['MasVnrType'].fillna(_input0['MasVnrType'].mode()[0])
_input0['BsmtCond'] = _input0['BsmtCond'].fillna(_input0['BsmtCond'].mode()[0])
_input0['BsmtExposure'] = _input0['BsmtExposure'].fillna(_input0['BsmtExposure'].mode()[0])
_input0['BsmtFinType1'] = _input0['BsmtFinType1'].fillna(_input0['BsmtFinType1'].mode()[0])
_input0['BsmtFinSF1'] = _input0['BsmtFinSF1'].fillna(_input0['BsmtFinSF1'].mean())
_input0['BsmtFinType2'] = _input0['BsmtFinType2'].fillna(_input0['BsmtFinType2'].mode()[0])
_input0['BsmtFinSF1'] = _input0['BsmtFinSF1'].fillna(_input0['BsmtFinSF1'].mean())
_input0['BsmtFinSF2'] = _input0['BsmtFinSF2'].fillna(_input0['BsmtFinSF2'].mean())
_input0['BsmtUnfSF'] = _input0['BsmtUnfSF'].fillna(_input0['BsmtUnfSF'].mean())
_input0['TotalBsmtSF'] = _input0['TotalBsmtSF'].fillna(_input0['TotalBsmtSF'].mean())
_input0['BsmtFullBath'] = _input0['BsmtFullBath'].fillna(_input0['BsmtFullBath'].mean())
_input0['BsmtHalfBath'] = _input0['BsmtHalfBath'].fillna(_input0['BsmtHalfBath'].mean())
_input0['KitchenQual'] = _input0['KitchenQual'].fillna(_input0['KitchenQual'].mode()[0])
_input0['Functional'] = _input0['Functional'].fillna(_input0['Functional'].mode()[0])
_input0['FireplaceQu'] = _input0['FireplaceQu'].fillna(_input0['FireplaceQu'].mode()[0])
_input0['GarageType'] = _input0['GarageType'].fillna(_input0['GarageType'].mode()[0])
_input0['GarageYrBlt'] = _input0['GarageYrBlt'].fillna(_input0['GarageYrBlt'].mean())
_input0['GarageFinish'] = _input0['GarageFinish'].fillna(_input0['GarageFinish'].mode()[0])
_input0['GarageCars'] = _input0['GarageCars'].fillna(_input0['GarageCars'].mean())
_input0['GarageArea'] = _input0['GarageArea'].fillna(_input0['GarageArea'].mean())
_input0['GarageQual'] = _input0['GarageQual'].fillna(_input0['GarageQual'].mode()[0])
_input0['GarageCond'] = _input0['GarageCond'].fillna(_input0['GarageCond'].mode()[0])
_input0['SaleType'] = _input0['SaleType'].fillna(_input0['SaleType'].mode()[0])
_input0 = _input0.drop(['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=False)
_input0.shape
_input0.isnull().sum().any()
corr = _input1.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr)
n = 10
cor_cols = corr.nlargest(n, 'SalePrice')['SalePrice'].index
cor_cols_1 = np.corrcoef(_input1[cor_cols].values.T)
sns.set(font_scale=1.5)
plt.figure(figsize=(16, 10))
sns.heatmap(cor_cols_1, cbar=True, annot=True, yticklabels=cor_cols.values, xticklabels=cor_cols.values)
cor_cols = pd.DataFrame(cor_cols)
cor_cols.columns = ['Top correlated features']
cor_cols
overall_qua = _input1['OverallQual'].value_counts().reset_index().rename(columns={'index': 'Rating', 'OverallQual': 'No. of customers'})
overall_qua
fig = px.pie(names=overall_qua['Rating'].values, values=overall_qua['No. of customers'].values, title='overall quality')
fig.show()
sns.set(style='whitegrid')
plt.figure(figsize=(15, 10))
fig = sns.barplot(x='OverallQual', y='SalePrice', data=_input1)
sns.set(style='whitegrid')
plt.figure(figsize=(15, 10))
fig = sns.scatterplot(x='GrLivArea', y='SalePrice', data=_input1)
_input1['GarageCars'].describe()
fig = px.box(_input1, x='GarageCars', y='SalePrice')
fig.show()
_input1['GarageArea'].describe()
fig = px.scatter(_input1, x='GarageArea', y='SalePrice')
fig.show()
plt.figure(figsize=(16, 7))
sns.scatterplot(x='TotalBsmtSF', y='SalePrice', data=_input1)
_input1['1stFlrSF'].describe()
fig = px.scatter(_input1, x='1stFlrSF', y='SalePrice')
fig.show()
_input1['FullBath'].value_counts()
plt.figure(figsize=(16, 7))
sns.boxplot(x=_input1['FullBath'], y=_input1['SalePrice'])
_input1['TotRmsAbvGrd'].value_counts()
plt.figure(figsize=(15, 10))
fig = sns.barplot(x='TotRmsAbvGrd', y='SalePrice', data=_input1)
_input1['YearBuilt'].value_counts().shape
plt.figure(figsize=(16, 7))
fig = px.bar(_input1, x='YearBuilt', y='SalePrice')
fig.show()
dataframe = pd.concat([_input1, _input0])
dataframe.shape
df = pd.get_dummies(dataframe, drop_first=True)
df.head()
train_df = df.iloc[:1460, :]
test_df = df.iloc[1460:, :]
X = train_df.drop('SalePrice', axis=1)
y = train_df['SalePrice']
test_df = test_df.drop('SalePrice', axis=1)
from sklearn.ensemble import RandomForestRegressor
forest_regressor = RandomForestRegressor()