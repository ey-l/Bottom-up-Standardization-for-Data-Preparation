import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from feature_engine import creation
from feature_engine import discretisation as disc
from feature_engine import encoding as enc
from feature_engine import imputation as imp
from feature_engine import selection as sel
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
(X_train, X_test, y_train, y_test) = train_test_split(_input1.drop(['Id', 'SalePrice'], axis=1), _input1['SalePrice'], test_size=0.05, random_state=0)
(X_train.shape, X_test.shape)
id_ = _input0['Id']
_input0 = _input0.drop('Id', axis=1, inplace=False)
_input0.shape

def create_master_data(train, test, submission, y_train, y_test):
    train = train.copy()
    test = test.copy()
    _input0 = _input0.copy()
    train['target'] = y_train
    train['data'] = 'train'
    train = train.reset_index(drop=True, inplace=False)
    test['target'] = y_test
    test['data'] = 'test'
    test = test.reset_index(drop=True, inplace=False)
    _input0['target'] = np.nan
    _input0['data'] = 'submission'
    _input0 = _input0.reset_index(drop=True, inplace=False)
    master_data = pd.concat([train, test, _input0], axis=0)
    master_data = master_data.reset_index(drop=True, inplace=False)
    return master_data
y_train.hist(bins=50, density=True)
y_test.hist(bins=50, density=True)
plt.legend(['Train', 'Test'])
plt.ylabel('Number of houses')
plt.xlabel('Sale Price')
np.log(y_train).hist(bins=50, density=True)
np.log(y_test).hist(bins=50, density=True)
plt.ylabel('Number of houses')
plt.xlabel('Log of Sale Price')
np.log(y_train).plot.density()
np.log(y_test).plot.density()
plt.xlabel('Log of Sale Price')
y_train = np.log(y_train)
y_test = np.log(y_test)
X_train['HasPool'] = np.where(X_train['PoolArea'] > 0, 1, 0)
X_test['HasPool'] = np.where(X_test['PoolArea'] > 0, 1, 0)
_input0['HasPool'] = np.where(_input0['PoolArea'] > 0, 1, 0)
X_train['HasAlley'] = np.where(X_train['Alley'].isnull(), 0, 1)
X_test['HasAlley'] = np.where(X_test['Alley'].isnull(), 0, 1)
_input0['HasAlley'] = np.where(_input0['Alley'].isnull(), 0, 1)

def plot_median_price(variables, limits):
    tmp = pd.concat([X_train[variables].reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    g = sns.PairGrid(tmp, x_vars=variables, y_vars=['SalePrice'])
    g.map(sns.barplot)
    plt.ylim(limits)
plot_median_price(['HasAlley', 'HasPool'], (11.5, 12.5))
X_train['Condition_total'] = np.where(X_train['Condition2'] == X_train['Condition1'], 0, 1)
X_test['Condition_total'] = np.where(X_test['Condition2'] == X_test['Condition1'], 0, 1)
_input0['Condition_total'] = np.where(_input0['Condition2'] == _input0['Condition1'], 0, 1)
X_train['Condition_total'].value_counts()
X_train['Exterior_total'] = np.where(X_train['Exterior1st'] == X_train['Exterior2nd'], 0, 1)
X_test['Exterior_total'] = np.where(X_test['Exterior1st'] == X_test['Exterior2nd'], 0, 1)
_input0['Exterior_total'] = np.where(_input0['Exterior1st'] == _input0['Exterior2nd'], 0, 1)
X_train['Exterior_total'].value_counts()
plot_median_price(['Condition_total', 'Exterior_total'], (11.5, 12.5))
categorical = [var for var in _input1.columns if _input1[var].dtype == 'O']
categorical = categorical + ['MSSubClass']
len(categorical)
X_train[categorical] = X_train[categorical].astype('O')
X_test[categorical] = X_test[categorical].astype('O')
_input0[categorical] = _input0[categorical].astype('O')
null_cat = {var: _input1[var].isnull().mean() for var in categorical if _input1[var].isnull().mean() > 0}
pd.Series(null_cat).sort_values().plot.bar(figsize=(10, 4))
plt.ylabel('Percentage of missing data')
plt.axhline(y=0.9, color='r', linestyle='-')
plt.axhline(y=0.8, color='g', linestyle='-')
tmp = pd.concat([X_train[['Alley', 'MiscFeature', 'PoolQC']].reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
for var in ['Alley', 'MiscFeature', 'PoolQC']:
    tmp[var] = np.where(tmp[var].isnull(), 1, 0)
g = sns.PairGrid(tmp, x_vars=['Alley', 'MiscFeature', 'PoolQC'], y_vars=['SalePrice'])
g.map(sns.barplot)
plt.ylim(10, 13)
drop_features = sel.DropFeatures(features_to_drop=['Alley', 'MiscFeature', 'PoolQC'])
X_train = drop_features.fit_transform(X_train)
X_test = drop_features.transform(X_test)
_input0 = drop_features.transform(_input0)
(X_train.shape, X_test.shape, _input0.shape)
cat_imputer = imp.CategoricalImputer(return_object=True)