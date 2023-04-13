import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn import tree
import pydotplus
import pandas as pd
import numpy as np
import collections
from math import sqrt
import scipy.stats as ss
from scipy import stats
from scipy.stats import norm, skew
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn import preprocessing, tree
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, LassoLarsCV, Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import plot_importance
from matplotlib import pyplot
import xgboost
pd.set_option('display.max_columns', None)
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
col_year = [feature for feature in _input1.columns if 'Year' in feature or 'Yr' in feature]
col_year_test = [feature for feature in _input0.columns if 'Year' in feature or 'Yr' in feature]
col_NonYr = [feature for feature in _input1.columns if feature not in col_year]
col_num = []
col_cat = []
for c in col_NonYr:
    if _input1[c].dtype != 'O' and c not in col_year:
        col_num.append(c)
    else:
        col_cat.append(c)
col_NonYr_test = [feature for feature in _input0.columns if feature not in col_year_test]
col_num_test = []
col_cat_test = []
for c in col_NonYr_test:
    if _input0[c].dtype != 'O' and c not in col_year_test:
        col_num_test.append(c)
    else:
        col_cat_test.append(c)
print('Training set Number of (rows,columns): ' + str(_input1.shape))
print('Testing set Number of (rows,columns): ' + str(_input0.shape))
_input1.describe()
print('Training data "SalePrice" skewness: %f' % _input1['SalePrice'].skew())
print('Training data "SalePrice" kurtosis: %f' % _input1['SalePrice'].kurt())
plt.figure(figsize=(6, 3))
sns.distplot(_input1['SalePrice'], fit=norm)
plt.figure(figsize=(6, 3))
res = stats.probplot(_input1['SalePrice'], plot=plt)
plt.figure(figsize=(6, 3))
_input1.boxplot(column=['SalePrice'])
data = _input1.copy()
data['SalePrice'] = np.log(data['SalePrice'])
print('Training data "SalePrice" skewness: %f' % data['SalePrice'].skew())
print('Training data "SalePrice" kurtosis: %f' % data['SalePrice'].kurt())
plt.figure(figsize=(6, 3))
sns.distplot(data['SalePrice'], fit=norm)
plt.figure(figsize=(6, 3))
res = stats.probplot(data['SalePrice'], plot=plt)
plt.figure(figsize=(6, 3))
data.boxplot(column=['SalePrice'])
print('Missing value:', _input1.isnull().sum().sum())
print('Duplicated rows:', _input1.duplicated().sum())
print('Duplicated columns:', _input1.columns.duplicated().sum())
col_train_with_missing = [c for c in _input1.columns if _input1[c].isnull().sum() >= 1]
col_test_with_missing = [c for c in _input0.columns if _input0[c].isnull().sum() >= 1]
print(f'Have missing values in testing but not in training data: {set(col_test_with_missing).difference(set(col_train_with_missing))}\n')
print(f'Have missing values in training but not in testing data: {set(col_train_with_missing).difference(set(col_test_with_missing))}')
plt.figure(figsize=(6, 3))
_input1[col_train_with_missing].isnull().mean().sort_values(ascending=False).plot.bar(ylabel='Missing value %', title='Missing value% of columns with missing values', color='cadetblue')
_input1[_input1[col_train_with_missing].isnull().any(axis=1)]
_input1[col_year].info()
fig = plt.figure(figsize=(30, 5))
for (count, feature) in enumerate(col_year, 1):
    data = _input1.copy()
    ax = fig.add_subplot(1, len(col_year), count)
    if feature == 'YrSold':
        ax.plot(data.groupby(feature)['SalePrice'].median(), color='fuchsia')
        ax.annotate('Price drops with yr sold makes no sense!', xy=[2008.0, 164000], xytext=[2006.0, 160000], arrowprops={'arrowstyle': '->', 'color': 'blue'})
    else:
        ax.plot(data.groupby(feature)['SalePrice'].median(), color='slategray')
    ax.set_xlabel(feature, fontsize='xx-large')
    ax.set_ylabel('SalePrice ($)')
sns.boxplot(data=_input1[col_year])
fig = plt.figure(figsize=(30, 5))
for (count, feature) in enumerate(['YearBuilt', 'YearRemodAdd', 'GarageYrBlt'], 1):
    data = _input1.copy()
    data[feature] = data['YrSold'] - data[feature]
    ax = fig.add_subplot(1, len(col_year), count)
    ax.plot(data.groupby(feature)['SalePrice'].median(), color='slategray')
    ax.set_xlabel('YearSold- ' + str(feature), fontsize='xx-large')
    ax.set_ylabel('SalePrice ($)')
print('There are {} numerical fields, {} categorical fields in training data'.format(len(col_num), len(col_cat)))
print('There are {} numerical fields, {} categorical fields in testing data'.format(len(col_num_test), len(col_cat_test)))
uniques = {}
for feature in col_num:
    unique = len(_input1[feature].unique())
    uniques[feature] = unique
uniques_sorted = {k: v for (k, v) in sorted(uniques.items(), key=lambda item: item[1])}
for pair in uniques_sorted.items():
    print(pair)
col_disc = [feature for feature in col_num if len(data[feature].unique()) <= 25]
col_cont = [feature for feature in col_num if feature not in col_disc and feature != 'Id']
col_disc_test = [feature for feature in col_num_test if len(_input0[feature].unique()) <= 25]
col_cont_test = [feature for feature in col_num_test if feature not in col_disc_test]
print(f'Discrete features: {col_disc}\n')
print(f'Continuous features: {col_cont}')
grh_per_row = 3
(fig, ax) = plt.subplots(len(col_cont) // grh_per_row + 1, grh_per_row, figsize=(30, 30))
for (count, feature) in enumerate(col_cont, 0):
    data = _input1.copy()
    row = count // grh_per_row
    col = count % grh_per_row
    ax[row, col].hist(_input1[feature], color='thistle')
    ax[row, col].set_xlabel(feature, fontsize='xx-large')
    ax[row, col].set_ylabel('Count')
grh_per_row = 3
(fig, ax) = plt.subplots(len(col_cont) // grh_per_row, grh_per_row, figsize=(40, 40))
for (count, feature) in enumerate(col_cont, 0):
    if feature == 'SalePrice':
        pass
    else:
        data = _input1.copy()
        row = count // grh_per_row
        col = count % grh_per_row
        ax[row, col].scatter(data[feature], data['SalePrice'], color='tan')
        ax[row, col].set_xlabel(feature, fontsize='xx-large')
        ax[row, col].set_ylabel('SalePrice ($)')
grh_per_row = 3
(fig, ax) = plt.subplots(len(col_disc) // grh_per_row + 1, grh_per_row, figsize=(30, 30))
for (count, feature) in enumerate(col_disc, 0):
    data = _input1.copy()
    row = count // grh_per_row
    col = count % grh_per_row
    ax[row, col].hist(_input1[feature], color='lightsteelblue', bins=data[feature].unique().sort())
    ax[row, col].set_xlabel(feature, fontsize='xx-large')
    ax[row, col].set_ylabel('Count')
grh_per_row = 3
(fig, ax) = plt.subplots(len(col_disc) // grh_per_row + 1, grh_per_row, figsize=(30, 30))
for (count, feature) in enumerate(col_disc, 0):
    data = _input1.copy()
    row = count // grh_per_row
    col = count % grh_per_row
    ax[row, col].scatter(data[feature], data['SalePrice'], color='cornflowerblue')
    ax[row, col].set_xlabel(feature, fontsize='xx-large')
    ax[row, col].set_ylabel('SalePrice ($)')
corr = _input1[col_num].corr()
plt.figure(figsize=(25, 23))
plt.title('Housing data numerical feature correlation')
sns.heatmap(data=corr, annot=True, cmap='BuPu')
grh_per_row = 3
(fig, ax) = plt.subplots(len(col_cat) // grh_per_row + 1, grh_per_row, figsize=(40, 120))
for (count, feature) in enumerate(col_cat, 0):
    data = _input1.copy()
    row = count // grh_per_row
    col = count % grh_per_row
    df = data.groupby(feature)['SalePrice'].mean().to_frame('SalePrice')
    ax[row, col].bar(df.index, df.SalePrice, color='burlywood')
    ax[row, col].set_xlabel(feature, fontsize='xx-large')
    ax[row, col].set_ylabel('SalePrice ($)')
    ax[row, col].set_xticklabels(df.index, fontsize='large', rotation=30)
for feature in col_cat:
    data = _input1.copy()
    print(feature)
    print(data[feature].value_counts())
    print('______________________')
for feature in col_num:
    if _input1[feature].isnull().sum() > 0:
        _input1[feature + '_nan'] = np.where(_input1[feature].isnull(), 1, 0)
        _input1[feature] = _input1[feature].fillna(_input1[feature].median(), inplace=False)
for feature in col_num_test:
    if _input0[feature].isnull().sum() > 0:
        _input0[feature + '_nan'] = np.where(_input0[feature].isnull(), 1, 0)
        _input0[feature] = _input0[feature].fillna(_input0[feature].median(), inplace=False)
print(_input1[col_num].isnull().sum().sum())
print(_input0[col_num_test].isnull().sum().sum())
for feature in col_cat:
    _input1[feature] = _input1[feature].fillna('Missing', inplace=False)
for feature in col_cat_test:
    _input0[feature] = _input0[feature].fillna('Missing', inplace=False)
print(_input1[col_cat].isnull().sum().sum())
print(_input0[col_cat_test].isnull().sum().sum())
for feature in col_year:
    if _input1[feature].isnull().sum() > 0:
        _input1[feature + '_nan'] = np.where(_input1[feature].isnull(), 1, 0)
        _input1[feature] = _input1[feature].fillna(_input1[feature].median(), inplace=False)
for feature in col_year_test:
    if _input0[feature].isnull().sum() > 0:
        _input0[feature + '_nan'] = np.where(_input0[feature].isnull(), 1, 0)
        _input0[feature] = _input0[feature].fillna(_input0[feature].median(), inplace=False)
print(_input1.isnull().sum().sum())
print(_input0.isnull().sum().sum())
for feature in col_year:
    if feature != 'YrSold':
        _input1['YrSold-' + feature] = np.maximum(_input1['YrSold'] - _input1[feature], 0)
for feature in col_year_test:
    if feature != 'YrSold':
        _input0['YrSold-' + feature] = np.maximum(_input0['YrSold'] - _input0[feature], 0)
for feature in col_cat:
    temp = _input1.groupby(feature)['SalePrice'].count() / len(_input1)
    temp_df = temp[temp > 0.01].index
    _input1[feature] = np.where(_input1[feature].isin(temp_df), _input1[feature], 'Rare_var')
    _input0[feature] = np.where(_input0[feature].isin(temp_df), _input0[feature], 'Rare_var')
for feature in col_cat:
    labels_ordered = _input1.groupby([feature])['SalePrice'].mean().sort_values().index
    labels_ordered = {k: i for (i, k) in enumerate(labels_ordered, 0)}
    _input1[feature] = _input1[feature].map(labels_ordered)
    test_default = collections.defaultdict(lambda : 0.0, labels_ordered)
    _input0[feature] = _input0[feature].map(test_default)
_input1['SalePrice'] = np.log1p(_input1['SalePrice'])
skewed_feats = _input1[col_num].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
_input1[skewed_feats] = np.log1p(_input1[skewed_feats])
_input0[skewed_feats] = np.log1p(_input0[skewed_feats])
_input1[col_cat] = np.log1p(_input1[col_cat])
_input0[col_cat] = np.log1p(_input0[col_cat])
grh_per_row = 4
(fig, ax) = plt.subplots(len(_input1.columns) // grh_per_row + 1, grh_per_row, figsize=(30, 100))
for (count, feature) in enumerate(_input1.columns, 0):
    data = _input1.copy()
    row = count // grh_per_row
    col = count % grh_per_row
    ax[row, col].scatter(data[feature], data['SalePrice'], color='mediumpurple')
    ax[row, col].set_xlabel(feature, fontsize='xx-large')
    ax[row, col].set_ylabel('SalePrice with log normalization ($)')
    ax[row, col].grid()
_input1 = _input1.drop(_input1[_input1['LotFrontage'] > 5].index)
_input1 = _input1.drop(_input1[_input1['LotArea'] > 11.5].index)
_input1 = _input1.drop(_input1[(_input1['YearBuilt'] < 1900) & (_input1['SalePrice'] > 12.4)].index)
_input1 = _input1.drop(_input1[(_input1['1stFlrSF'] > 8) & (_input1['SalePrice'] < 12.5)].index)
_input1.shape
y_train = _input1['SalePrice']
X_train = _input1.drop(['SalePrice', 'Id'], axis=1)
X_test = _input0.drop(['Id'], axis=1)

def get_score(model, scaler, alpha):
    my_pipeline = Pipeline(steps=[('My scaler', scaler()), ('My classifier', model(alpha=alpha))])
    scores = -1 * cross_val_score(my_pipeline, X_train, y_train, cv=3, scoring='neg_mean_squared_error')
    return scores.mean()

def model_scores(model, scaler, alphas):
    results = {}
    best_score = float('inf')
    best_alpha = 0
    for alpha in alphas:
        score = get_score(model=model, scaler=scaler, alpha=alpha)
        if score < best_score:
            best_score = score
            best_alpha = alpha
        results[alpha] = score
    print(f'\nBest alpha: {best_alpha} with score of {best_score}')
    (fig, ax) = plt.subplots(figsize=(10, 5))
    ax.plot(list(results.keys()), list(results.values()), markersize=5, marker='o', color='royalblue')
    ax.set_title(str(model) + str(scaler))
    ax.set_xlabel('alpha')
    ax.set_ylabel('MSE score')
    return best_score

def draw_results_lasso(model, scaler, alpha):
    (X_cv, X_test_train, y_cv, y_test_train) = train_test_split(X_train, y_train, test_size=0.2, random_state=123)
    my_pipeline = Pipeline(steps=[('My scaler', scaler()), ('My classifier', model(alpha=alpha))])