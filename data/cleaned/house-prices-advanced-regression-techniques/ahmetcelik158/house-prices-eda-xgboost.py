import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy import stats

train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train.head()
print('Columns:')
train.columns
sns.displot(train['SalePrice'], kde=True, aspect=1.5)

print('\nSALE PRICE SUMMARY: ')
train['SalePrice'].describe()
train['SalePrice_Log'] = np.log(train['SalePrice'])
(fig, ax) = plt.subplots(2, 2, figsize=(12, 6))
sns.histplot(ax=ax[0, 0], data=train, x='SalePrice', kde=True)
res = stats.probplot(train['SalePrice'], plot=ax[0, 1])
ax[0, 0].set_title('Sale Price Distribution (Before Log Transform)')
ax[0, 1].set_title('Probability Plot (Before Log Transform)')
sns.histplot(ax=ax[1, 0], data=train, x='SalePrice_Log', kde=True)
res = stats.probplot(train['SalePrice_Log'], plot=ax[1, 1])
ax[1, 0].set_title('Sale Price Distribution (After Log Transform)')
ax[1, 1].set_title('Probability Plot (After Log Transform)')
fig.tight_layout()

print('\nSale Price After Log Transform: ')
train['SalePrice_Log'].describe()
corr_matrix = train.corr()
plt.figure(figsize=(9, 8))
sns.heatmap(corr_matrix)
top_cols = corr_matrix.sort_values(by='SalePrice_Log', ascending=False).head(15).index
top_corr_matrix = train[top_cols].corr()
plt.figure(figsize=(9, 8))
sns.heatmap(top_corr_matrix, annot=True, square=True)
vars_to_drop = list()
vars_log_transform = list()
sns.stripplot(x='OverallQual', y='SalePrice_Log', data=train)
cols = ['SalePrice', 'GrLivArea', 'TotRmsAbvGrd', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF']
sns.pairplot(data=train[cols])
vars_log_transform.append('GrLivArea')
vars_log_transform.append('1stFlrSF')
vars_to_drop.append('TotRmsAbvGrd')
vars_to_drop.append('2ndFlrSF')
vars_to_drop.append('LowQualFinSF')
cols = ['SalePrice', 'TotalBsmtSF', '1stFlrSF', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF']
sns.pairplot(data=train[cols])
train['HasBsmt'] = 0
train.loc[train['TotalBsmtSF'] > 0, 'HasBsmt'] = 1
test['HasBsmt'] = 0
test.loc[test['TotalBsmtSF'] > 0, 'HasBsmt'] = 1
vars_to_drop.append('TotalBsmtSF')
sns.boxplot(x='HasBsmt', y='SalePrice_Log', data=train)
bsmt_vars = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
train[bsmt_vars] = train[bsmt_vars].fillna(value='NoBsmt')
test[bsmt_vars] = test[bsmt_vars].fillna(value='NoBsmt')
(fig, ax) = plt.subplots(2, 3, figsize=(12, 6))
sns.stripplot(ax=ax[0, 0], x='BsmtQual', y='SalePrice_Log', data=train)
sns.stripplot(ax=ax[0, 1], x='BsmtCond', y='SalePrice_Log', data=train)
sns.stripplot(ax=ax[0, 2], x='BsmtExposure', y='SalePrice_Log', data=train)
sns.stripplot(ax=ax[1, 0], x='BsmtFinType1', y='SalePrice_Log', data=train)
sns.stripplot(ax=ax[1, 1], x='BsmtFinType2', y='SalePrice_Log', data=train)
ax[1, 2].remove()
fig.tight_layout()

cols = ['SalePrice', 'GarageCars', 'GarageArea']
sns.pairplot(data=train[cols])
vars_to_drop.append('GarageArea')
garage_vars = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
train[garage_vars] = train[garage_vars].fillna(value='NoGrg')
test[garage_vars] = test[garage_vars].fillna(value='NoGrg')
(fig, ax) = plt.subplots(2, 2, figsize=(12, 6))
sns.stripplot(ax=ax[0, 0], x='GarageType', y='SalePrice_Log', data=train)
sns.stripplot(ax=ax[0, 1], x='GarageFinish', y='SalePrice_Log', data=train)
sns.stripplot(ax=ax[1, 0], x='GarageQual', y='SalePrice_Log', data=train)
sns.stripplot(ax=ax[1, 1], x='GarageCond', y='SalePrice_Log', data=train)
fig.tight_layout()

to_replace = ['Basment', 'CarPort', '2Types']
train['GarageType'] = train['GarageType'].replace(to_replace, 'Other')
test['GarageType'] = test['GarageType'].replace(to_replace, 'Other')
test[['TotalBsmtSF', 'GarageArea']] = test[['TotalBsmtSF', 'GarageArea']].fillna(value=0)
train['TotalArea'] = train['GrLivArea'] + train['TotalBsmtSF'] + train['GarageArea']
test['TotalArea'] = test['GrLivArea'] + test['TotalBsmtSF'] + test['GarageArea']
train['QualArea'] = train['TotalArea'] * train['OverallQual']
test['QualArea'] = test['TotalArea'] * test['OverallQual']
vars_log_transform.append('TotalArea')
vars_log_transform.append('QualArea')
vars_log_transform.append('LotArea')
(fig, ax) = plt.subplots(1, 2, figsize=(12, 3.5))
sns.scatterplot(ax=ax[0], x='TotalArea', y='SalePrice', data=train)
sns.scatterplot(ax=ax[1], x='QualArea', y='SalePrice', data=train)
fig.tight_layout()

row_ind = train[(train['TotalArea'] > 8000) & (train['SalePrice'] < 300000)].index
train = train.drop(row_ind, axis=0)
(fig, ax) = plt.subplots(1, 2, figsize=(12, 3.5))
sns.scatterplot(ax=ax[0], x='TotalArea', y='SalePrice', data=train)
sns.scatterplot(ax=ax[1], x='QualArea', y='SalePrice', data=train)
fig.tight_layout()

cols = ['SalePrice_Log', 'YearBuilt', 'GarageYrBlt', 'YearRemodAdd']
sns.pairplot(data=train[cols])
(fig, ax) = plt.subplots(1, 2, figsize=(12, 4))
sns.regplot(ax=ax[0], x='MoSold', y='SalePrice_Log', data=train)
sns.regplot(ax=ax[1], x='YrSold', y='SalePrice_Log', data=train)
fig.tight_layout()

vars_to_drop.append('GarageYrBlt')
vars_to_drop.append('MoSold')
vars_to_drop.append('YrSold')
(fig, ax) = plt.subplots(1, 2, figsize=(12, 3.5))
sns.scatterplot(ax=ax[0], x='PoolArea', y='SalePrice_Log', data=train)
sns.stripplot(ax=ax[1], x='PoolQC', y='SalePrice_Log', data=train)
fig.tight_layout()

train['HasPool'] = 0
train.loc[train['PoolArea'] > 0, 'HasPool'] = 1
test['HasPool'] = 0
test.loc[test['PoolArea'] > 0, 'HasPool'] = 1
vars_to_drop.append('PoolArea')
vars_to_drop.append('PoolQC')
(fig, ax) = plt.subplots(1, 4, figsize=(12, 3))
sns.regplot(ax=ax[0], x='FullBath', y='SalePrice_Log', data=train)
sns.regplot(ax=ax[1], x='BsmtFullBath', y='SalePrice_Log', data=train)
sns.regplot(ax=ax[2], x='HalfBath', y='SalePrice_Log', data=train)
sns.regplot(ax=ax[3], x='BsmtHalfBath', y='SalePrice_Log', data=train)
fig.tight_layout()

test[['BsmtFullBath', 'BsmtHalfBath']] = test[['BsmtFullBath', 'BsmtHalfBath']].fillna(value=0)
train['TotalBath'] = train['FullBath'] + train['BsmtFullBath'] + 0.5 * (train['HalfBath'] + train['BsmtHalfBath'])
test['TotalBath'] = test['FullBath'] + test['BsmtFullBath'] + 0.5 * (test['HalfBath'] + test['BsmtHalfBath'])
vars_to_drop.append('BsmtHalfBath')
sns.regplot(x='TotalBath', y='SalePrice_Log', data=train)
train['TotalPorch'] = train['WoodDeckSF'] + train['OpenPorchSF'] + train['EnclosedPorch'] + train['3SsnPorch'] + train['ScreenPorch']
test['TotalPorch'] = test['WoodDeckSF'] + test['OpenPorchSF'] + test['EnclosedPorch'] + test['3SsnPorch'] + test['ScreenPorch']
(fig, ax) = plt.subplots(2, 3, figsize=(12, 6))
sns.regplot(ax=ax[0, 0], x='WoodDeckSF', y='SalePrice_Log', data=train)
sns.regplot(ax=ax[0, 1], x='OpenPorchSF', y='SalePrice_Log', data=train)
sns.regplot(ax=ax[0, 2], x='EnclosedPorch', y='SalePrice_Log', data=train)
sns.regplot(ax=ax[1, 0], x='3SsnPorch', y='SalePrice_Log', data=train)
sns.regplot(ax=ax[1, 1], x='ScreenPorch', y='SalePrice_Log', data=train)
sns.regplot(ax=ax[1, 2], x='TotalPorch', y='SalePrice_Log', data=train)
fig.tight_layout()

train['MSSubClass'] = train['MSSubClass'].astype(str)
test['MSSubClass'] = test['MSSubClass'].astype(str)
sns.stripplot(x='MSSubClass', y='SalePrice_Log', data=train)

(fig, ax) = plt.subplots(2, 3, figsize=(12, 6))
sns.scatterplot(ax=ax[0, 0], x='MiscVal', y='SalePrice_Log', data=train)
sns.stripplot(ax=ax[0, 1], x='Street', y='SalePrice_Log', data=train)
sns.stripplot(ax=ax[0, 2], x='Utilities', y='SalePrice_Log', data=train)
sns.stripplot(ax=ax[1, 0], x='Condition2', y='SalePrice_Log', data=train)
sns.stripplot(ax=ax[1, 1], x='RoofMatl', y='SalePrice_Log', data=train)
sns.stripplot(ax=ax[1, 2], x='Heating', y='SalePrice_Log', data=train)
fig.tight_layout()

vars_to_drop.append('MiscVal')
vars_to_drop.append('Street')
vars_to_drop.append('Utilities')
vars_to_drop.append('Condition2')
vars_to_drop.append('RoofMatl')
vars_to_drop.append('Heating')
for col in vars_log_transform:
    train[col] = np.log(train[col])
    test[col] = np.log(test[col])
print('Log transform is applied to following variables:')
print(', '.join(vars_log_transform))
train = train.drop(vars_to_drop, axis=1)
test = test.drop(vars_to_drop, axis=1)
print('Following variables are dropped:')
print(', '.join(vars_to_drop))
train = train.drop('SalePrice', axis=1)
train_miss = train.isnull().sum().sort_values(ascending=False).to_frame('N_MissVal')
cat_cols = train.select_dtypes(include=['object']).columns
train_miss['ValType'] = np.where(train_miss.index.isin(cat_cols), 'Categorical', 'Numerical')
print('Training Data:')
train_miss[train_miss['N_MissVal'] > 0]
cols = ['FireplaceQu', 'Fence', 'Alley', 'MiscFeature']
train[cols] = train[cols].fillna(value='None')
test[cols] = test[cols].fillna(value='None')
to_drop = train_miss[train_miss['N_MissVal'] > 1].index
to_drop = [x for x in to_drop if x not in cols]
train = train.drop(to_drop, axis=1)
test = test.drop(to_drop, axis=1)
row_ind = train.loc[train['Electrical'].isnull()].index
train = train.drop(row_ind, axis=0)
test_miss = test.isnull().sum().sort_values(ascending=False).to_frame('N_MissVal')
cat_cols = test.select_dtypes(include=['object']).columns
test_miss['ValType'] = np.where(test_miss.index.isin(cat_cols), 'Categorical', 'Numerical')
print('Test Data:')
test_miss[test_miss['N_MissVal'] > 0]
print('Missing MSZoning in test data')
print(test.loc[test.MSZoning.isnull(), ['MSZoning', 'Neighborhood']])
print('\n\nMSZoning Value Counts for IDOTRR:')
print(test.loc[test.Neighborhood == 'IDOTRR', ['MSZoning']].value_counts())
print('\n\nMSZoning Value Counts for Mitchel:')
print(test.loc[test.Neighborhood == 'Mitchel', ['MSZoning']].value_counts())
test.loc[test.MSZoning.isnull() & (test.Neighborhood == 'IDOTRR'), 'MSZoning'] = 'RM'
test.loc[test.MSZoning.isnull() & (test.Neighborhood == 'Mitchel'), 'MSZoning'] = 'RL'
cols = ['GarageCars', 'BsmtUnfSF', 'BsmtFinSF1', 'BsmtFinSF2']
test[cols] = test[cols].fillna(value=0)
test = test.apply(lambda x: x.fillna(x.value_counts().index[0]))
print('Remaining missing values (train): {}'.format(train.isnull().sum().sum()))
print('Remaining missing values (test): {}'.format(test.isnull().sum().sum()))
train.columns
print(f'Train data, shape before conversion: {train.shape}')
print(f'Test data, shape before conversion:  {test.shape}')
train = pd.get_dummies(train)
test = pd.get_dummies(test)
print(f'Train data, shape after conversion: {train.shape}')
print(f'Test data, shape after conversion:  {test.shape}')
cols_to_create = list(set(train.columns) - set(test.columns))
test[cols_to_create] = 0
test = test[train.columns]
test = test.drop('SalePrice_Log', axis=1)
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
X_train = train.drop(['SalePrice_Log', 'Id'], axis=1)
X_test = test.drop(['Id'], axis=1)
X_train = X_train.values
X_test = X_test.values
y_train = train['SalePrice_Log'].values
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