import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy import stats
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
print('Columns:')
_input1.columns
sns.displot(_input1['SalePrice'], kde=True, aspect=1.5)
print('\nSALE PRICE SUMMARY: ')
_input1['SalePrice'].describe()
_input1['SalePrice_Log'] = np.log(_input1['SalePrice'])
(fig, ax) = plt.subplots(2, 2, figsize=(12, 6))
sns.histplot(ax=ax[0, 0], data=_input1, x='SalePrice', kde=True)
res = stats.probplot(_input1['SalePrice'], plot=ax[0, 1])
ax[0, 0].set_title('Sale Price Distribution (Before Log Transform)')
ax[0, 1].set_title('Probability Plot (Before Log Transform)')
sns.histplot(ax=ax[1, 0], data=_input1, x='SalePrice_Log', kde=True)
res = stats.probplot(_input1['SalePrice_Log'], plot=ax[1, 1])
ax[1, 0].set_title('Sale Price Distribution (After Log Transform)')
ax[1, 1].set_title('Probability Plot (After Log Transform)')
fig.tight_layout()
print('\nSale Price After Log Transform: ')
_input1['SalePrice_Log'].describe()
corr_matrix = _input1.corr()
plt.figure(figsize=(9, 8))
sns.heatmap(corr_matrix)
top_cols = corr_matrix.sort_values(by='SalePrice_Log', ascending=False).head(15).index
top_corr_matrix = _input1[top_cols].corr()
plt.figure(figsize=(9, 8))
sns.heatmap(top_corr_matrix, annot=True, square=True)
vars_to_drop = list()
vars_log_transform = list()
sns.stripplot(x='OverallQual', y='SalePrice_Log', data=_input1)
cols = ['SalePrice', 'GrLivArea', 'TotRmsAbvGrd', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF']
sns.pairplot(data=_input1[cols])
vars_log_transform.append('GrLivArea')
vars_log_transform.append('1stFlrSF')
vars_to_drop.append('TotRmsAbvGrd')
vars_to_drop.append('2ndFlrSF')
vars_to_drop.append('LowQualFinSF')
cols = ['SalePrice', 'TotalBsmtSF', '1stFlrSF', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF']
sns.pairplot(data=_input1[cols])
_input1['HasBsmt'] = 0
_input1.loc[_input1['TotalBsmtSF'] > 0, 'HasBsmt'] = 1
_input0['HasBsmt'] = 0
_input0.loc[_input0['TotalBsmtSF'] > 0, 'HasBsmt'] = 1
vars_to_drop.append('TotalBsmtSF')
sns.boxplot(x='HasBsmt', y='SalePrice_Log', data=_input1)
bsmt_vars = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
_input1[bsmt_vars] = _input1[bsmt_vars].fillna(value='NoBsmt')
_input0[bsmt_vars] = _input0[bsmt_vars].fillna(value='NoBsmt')
(fig, ax) = plt.subplots(2, 3, figsize=(12, 6))
sns.stripplot(ax=ax[0, 0], x='BsmtQual', y='SalePrice_Log', data=_input1)
sns.stripplot(ax=ax[0, 1], x='BsmtCond', y='SalePrice_Log', data=_input1)
sns.stripplot(ax=ax[0, 2], x='BsmtExposure', y='SalePrice_Log', data=_input1)
sns.stripplot(ax=ax[1, 0], x='BsmtFinType1', y='SalePrice_Log', data=_input1)
sns.stripplot(ax=ax[1, 1], x='BsmtFinType2', y='SalePrice_Log', data=_input1)
ax[1, 2].remove()
fig.tight_layout()
cols = ['SalePrice', 'GarageCars', 'GarageArea']
sns.pairplot(data=_input1[cols])
vars_to_drop.append('GarageArea')
garage_vars = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
_input1[garage_vars] = _input1[garage_vars].fillna(value='NoGrg')
_input0[garage_vars] = _input0[garage_vars].fillna(value='NoGrg')
(fig, ax) = plt.subplots(2, 2, figsize=(12, 6))
sns.stripplot(ax=ax[0, 0], x='GarageType', y='SalePrice_Log', data=_input1)
sns.stripplot(ax=ax[0, 1], x='GarageFinish', y='SalePrice_Log', data=_input1)
sns.stripplot(ax=ax[1, 0], x='GarageQual', y='SalePrice_Log', data=_input1)
sns.stripplot(ax=ax[1, 1], x='GarageCond', y='SalePrice_Log', data=_input1)
fig.tight_layout()
to_replace = ['Basment', 'CarPort', '2Types']
_input1['GarageType'] = _input1['GarageType'].replace(to_replace, 'Other')
_input0['GarageType'] = _input0['GarageType'].replace(to_replace, 'Other')
_input0[['TotalBsmtSF', 'GarageArea']] = _input0[['TotalBsmtSF', 'GarageArea']].fillna(value=0)
_input1['TotalArea'] = _input1['GrLivArea'] + _input1['TotalBsmtSF'] + _input1['GarageArea']
_input0['TotalArea'] = _input0['GrLivArea'] + _input0['TotalBsmtSF'] + _input0['GarageArea']
_input1['QualArea'] = _input1['TotalArea'] * _input1['OverallQual']
_input0['QualArea'] = _input0['TotalArea'] * _input0['OverallQual']
vars_log_transform.append('TotalArea')
vars_log_transform.append('QualArea')
vars_log_transform.append('LotArea')
(fig, ax) = plt.subplots(1, 2, figsize=(12, 3.5))
sns.scatterplot(ax=ax[0], x='TotalArea', y='SalePrice', data=_input1)
sns.scatterplot(ax=ax[1], x='QualArea', y='SalePrice', data=_input1)
fig.tight_layout()
row_ind = _input1[(_input1['TotalArea'] > 8000) & (_input1['SalePrice'] < 300000)].index
_input1 = _input1.drop(row_ind, axis=0)
(fig, ax) = plt.subplots(1, 2, figsize=(12, 3.5))
sns.scatterplot(ax=ax[0], x='TotalArea', y='SalePrice', data=_input1)
sns.scatterplot(ax=ax[1], x='QualArea', y='SalePrice', data=_input1)
fig.tight_layout()
cols = ['SalePrice_Log', 'YearBuilt', 'GarageYrBlt', 'YearRemodAdd']
sns.pairplot(data=_input1[cols])
(fig, ax) = plt.subplots(1, 2, figsize=(12, 4))
sns.regplot(ax=ax[0], x='MoSold', y='SalePrice_Log', data=_input1)
sns.regplot(ax=ax[1], x='YrSold', y='SalePrice_Log', data=_input1)
fig.tight_layout()
vars_to_drop.append('GarageYrBlt')
vars_to_drop.append('MoSold')
vars_to_drop.append('YrSold')
(fig, ax) = plt.subplots(1, 2, figsize=(12, 3.5))
sns.scatterplot(ax=ax[0], x='PoolArea', y='SalePrice_Log', data=_input1)
sns.stripplot(ax=ax[1], x='PoolQC', y='SalePrice_Log', data=_input1)
fig.tight_layout()
_input1['HasPool'] = 0
_input1.loc[_input1['PoolArea'] > 0, 'HasPool'] = 1
_input0['HasPool'] = 0
_input0.loc[_input0['PoolArea'] > 0, 'HasPool'] = 1
vars_to_drop.append('PoolArea')
vars_to_drop.append('PoolQC')
(fig, ax) = plt.subplots(1, 4, figsize=(12, 3))
sns.regplot(ax=ax[0], x='FullBath', y='SalePrice_Log', data=_input1)
sns.regplot(ax=ax[1], x='BsmtFullBath', y='SalePrice_Log', data=_input1)
sns.regplot(ax=ax[2], x='HalfBath', y='SalePrice_Log', data=_input1)
sns.regplot(ax=ax[3], x='BsmtHalfBath', y='SalePrice_Log', data=_input1)
fig.tight_layout()
_input0[['BsmtFullBath', 'BsmtHalfBath']] = _input0[['BsmtFullBath', 'BsmtHalfBath']].fillna(value=0)
_input1['TotalBath'] = _input1['FullBath'] + _input1['BsmtFullBath'] + 0.5 * (_input1['HalfBath'] + _input1['BsmtHalfBath'])
_input0['TotalBath'] = _input0['FullBath'] + _input0['BsmtFullBath'] + 0.5 * (_input0['HalfBath'] + _input0['BsmtHalfBath'])
vars_to_drop.append('BsmtHalfBath')
sns.regplot(x='TotalBath', y='SalePrice_Log', data=_input1)
_input1['TotalPorch'] = _input1['WoodDeckSF'] + _input1['OpenPorchSF'] + _input1['EnclosedPorch'] + _input1['3SsnPorch'] + _input1['ScreenPorch']
_input0['TotalPorch'] = _input0['WoodDeckSF'] + _input0['OpenPorchSF'] + _input0['EnclosedPorch'] + _input0['3SsnPorch'] + _input0['ScreenPorch']
(fig, ax) = plt.subplots(2, 3, figsize=(12, 6))
sns.regplot(ax=ax[0, 0], x='WoodDeckSF', y='SalePrice_Log', data=_input1)
sns.regplot(ax=ax[0, 1], x='OpenPorchSF', y='SalePrice_Log', data=_input1)
sns.regplot(ax=ax[0, 2], x='EnclosedPorch', y='SalePrice_Log', data=_input1)
sns.regplot(ax=ax[1, 0], x='3SsnPorch', y='SalePrice_Log', data=_input1)
sns.regplot(ax=ax[1, 1], x='ScreenPorch', y='SalePrice_Log', data=_input1)
sns.regplot(ax=ax[1, 2], x='TotalPorch', y='SalePrice_Log', data=_input1)
fig.tight_layout()
_input1['MSSubClass'] = _input1['MSSubClass'].astype(str)
_input0['MSSubClass'] = _input0['MSSubClass'].astype(str)
sns.stripplot(x='MSSubClass', y='SalePrice_Log', data=_input1)
(fig, ax) = plt.subplots(2, 3, figsize=(12, 6))
sns.scatterplot(ax=ax[0, 0], x='MiscVal', y='SalePrice_Log', data=_input1)
sns.stripplot(ax=ax[0, 1], x='Street', y='SalePrice_Log', data=_input1)
sns.stripplot(ax=ax[0, 2], x='Utilities', y='SalePrice_Log', data=_input1)
sns.stripplot(ax=ax[1, 0], x='Condition2', y='SalePrice_Log', data=_input1)
sns.stripplot(ax=ax[1, 1], x='RoofMatl', y='SalePrice_Log', data=_input1)
sns.stripplot(ax=ax[1, 2], x='Heating', y='SalePrice_Log', data=_input1)
fig.tight_layout()
vars_to_drop.append('MiscVal')
vars_to_drop.append('Street')
vars_to_drop.append('Utilities')
vars_to_drop.append('Condition2')
vars_to_drop.append('RoofMatl')
vars_to_drop.append('Heating')
for col in vars_log_transform:
    _input1[col] = np.log(_input1[col])
    _input0[col] = np.log(_input0[col])
print('Log transform is applied to following variables:')
print(', '.join(vars_log_transform))
_input1 = _input1.drop(vars_to_drop, axis=1)
_input0 = _input0.drop(vars_to_drop, axis=1)
print('Following variables are dropped:')
print(', '.join(vars_to_drop))
_input1 = _input1.drop('SalePrice', axis=1)
train_miss = _input1.isnull().sum().sort_values(ascending=False).to_frame('N_MissVal')
cat_cols = _input1.select_dtypes(include=['object']).columns
train_miss['ValType'] = np.where(train_miss.index.isin(cat_cols), 'Categorical', 'Numerical')
print('Training Data:')
train_miss[train_miss['N_MissVal'] > 0]
cols = ['FireplaceQu', 'Fence', 'Alley', 'MiscFeature']
_input1[cols] = _input1[cols].fillna(value='None')
_input0[cols] = _input0[cols].fillna(value='None')
to_drop = train_miss[train_miss['N_MissVal'] > 1].index
to_drop = [x for x in to_drop if x not in cols]
_input1 = _input1.drop(to_drop, axis=1)
_input0 = _input0.drop(to_drop, axis=1)
row_ind = _input1.loc[_input1['Electrical'].isnull()].index
_input1 = _input1.drop(row_ind, axis=0)
test_miss = _input0.isnull().sum().sort_values(ascending=False).to_frame('N_MissVal')
cat_cols = _input0.select_dtypes(include=['object']).columns
test_miss['ValType'] = np.where(test_miss.index.isin(cat_cols), 'Categorical', 'Numerical')
print('Test Data:')
test_miss[test_miss['N_MissVal'] > 0]
print('Missing MSZoning in test data')
print(_input0.loc[_input0.MSZoning.isnull(), ['MSZoning', 'Neighborhood']])
print('\n\nMSZoning Value Counts for IDOTRR:')
print(_input0.loc[_input0.Neighborhood == 'IDOTRR', ['MSZoning']].value_counts())
print('\n\nMSZoning Value Counts for Mitchel:')
print(_input0.loc[_input0.Neighborhood == 'Mitchel', ['MSZoning']].value_counts())
_input0.loc[_input0.MSZoning.isnull() & (_input0.Neighborhood == 'IDOTRR'), 'MSZoning'] = 'RM'
_input0.loc[_input0.MSZoning.isnull() & (_input0.Neighborhood == 'Mitchel'), 'MSZoning'] = 'RL'
cols = ['GarageCars', 'BsmtUnfSF', 'BsmtFinSF1', 'BsmtFinSF2']
_input0[cols] = _input0[cols].fillna(value=0)
_input0 = _input0.apply(lambda x: x.fillna(x.value_counts().index[0]))
print('Remaining missing values (train): {}'.format(_input1.isnull().sum().sum()))
print('Remaining missing values (test): {}'.format(_input0.isnull().sum().sum()))
_input1.columns
print(f'Train data, shape before conversion: {_input1.shape}')
print(f'Test data, shape before conversion:  {_input0.shape}')
_input1 = pd.get_dummies(_input1)
_input0 = pd.get_dummies(_input0)
print(f'Train data, shape after conversion: {_input1.shape}')
print(f'Test data, shape after conversion:  {_input0.shape}')
cols_to_create = list(set(_input1.columns) - set(_input0.columns))
_input0[cols_to_create] = 0
_input0 = _input0[_input1.columns]
_input0 = _input0.drop('SalePrice_Log', axis=1)
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error

def print_rmse(regressor, X, y, cv=5):
    neg_mse = cross_val_score(estimator=regressor, X=X, y=y, cv=cv, scoring='neg_mean_squared_error')
    rmse = np.sqrt(-neg_mse).mean()
    print(f'Regressor: {regressor}')
    print(f'RMSE: {rmse:.5f}\n')
    return
X_train = _input1.drop(['SalePrice_Log', 'Id'], axis=1)
X_test = _input0.drop(['Id'], axis=1)
X_train = X_train.values
X_test = X_test.values
y_train = _input1['SalePrice_Log'].values
regressor = LinearRegression()
print_rmse(regressor, X_train, y_train)
regressor = SVR()
print_rmse(regressor, X_train, y_train)
regressor = DecisionTreeRegressor(random_state=42)
print_rmse(regressor, X_train, y_train)
regressor = RandomForestRegressor(random_state=42)
print_rmse(regressor, X_train, y_train)
parameters = {'n_estimators': [100, 200, 500, 1000], 'max_depth': [20, 40, 100, None], 'min_samples_split': [2, 3, 5], 'min_samples_leaf': [1, 2, 4], 'max_features': [1, 'sqrt', 'log2']}
grid = RandomizedSearchCV(RandomForestRegressor(random_state=42), parameters, n_iter=12, refit=True, scoring='neg_mean_squared_error', random_state=42)