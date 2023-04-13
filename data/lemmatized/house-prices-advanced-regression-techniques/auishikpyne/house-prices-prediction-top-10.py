import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from xgboost import XGBRegressor
import sklearn.metrics as metrics
import math
from scipy.stats import norm, skew
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
(_input1.shape, _input0.shape)
_input1.head()
_input1.info()
print(_input1['SalePrice'].describe())
sns.distplot(_input1['SalePrice'])
_input1['SalePrice'] = np.log1p(_input1['SalePrice'])
sns.distplot(_input1['SalePrice'], fit=norm)
corrmat = _input1.corr()
(f, ax) = plt.subplots(figsize=(20, 12))
sns.heatmap(corrmat, vmax=0.8, square=True)
corr = _input1.corr()
highest_corr_features = corr.index[abs(corr['SalePrice']) > 0.5]
plt.figure(figsize=(10, 10))
g = sns.heatmap(_input1[highest_corr_features].corr(), annot=True, cmap='RdYlGn')
corr['SalePrice'].sort_values(ascending=False)
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(_input1[cols])
y_train = _input1['SalePrice']
test_id = _input0['Id']
all_data = pd.concat([_input1, _input0], axis=0, sort=False)
all_data = all_data.drop(['Id', 'SalePrice'], axis=1)
Total = all_data.isnull().sum().sort_values(ascending=False)
percent = (all_data.isnull().sum() / all_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([Total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(25)
all_data = all_data.drop(missing_data[missing_data['Total'] > 5].index, axis=1, inplace=False)
print(all_data.isnull().sum().max())
total = all_data.isnull().sum().sort_values(ascending=False)
total.head(19)
numeric_missed = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageArea', 'GarageCars']
for feature in numeric_missed:
    all_data[feature] = all_data[feature].fillna(0)
categorical_missed = ['Exterior1st', 'Exterior2nd', 'SaleType', 'MSZoning', 'Electrical', 'KitchenQual']
for feature in categorical_missed:
    all_data[feature] = all_data[feature].fillna(all_data[feature].mode()[0])
all_data['Functional'] = all_data['Functional'].fillna('Typ')
all_data = all_data.drop(['Utilities'], axis=1, inplace=False)
all_data.isnull().sum().max()
numeric_feats = all_data.dtypes[all_data.dtypes != 'object'].index
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)
high_skew = skewed_feats[abs(skewed_feats) > 0.5]
high_skew
for feature in high_skew.index:
    all_data[feature] = np.log1p(all_data[feature])
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data = pd.get_dummies(all_data)
all_data.head()
x_train = all_data[:len(y_train)]
x_test = all_data[len(y_train):]
(x_test.shape, x_train.shape)
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
scorer = make_scorer(mean_squared_error, greater_is_better=False)

def rmse_CV_train(model):
    kf = KFold(5, shuffle=True, random_state=42).get_n_splits(x_train.values)
    rmse = np.sqrt(-cross_val_score(model, x_train, y_train, scoring='neg_mean_squared_error', cv=kf))
    return rmse

def rmse_CV_test(model):
    kf = KFold(5, shuffle=True, random_state=42).get_n_splits(_input1.values)
    rmse = np.sqrt(-cross_val_score(model, x_test, y_test, scoring='neg_mean_squared_error', cv=kf))
    return rmse
import xgboost as XGB
the_model = XGB.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, learning_rate=0.05, max_depth=3, min_child_weight=1.7817, n_estimators=2200, reg_alpha=0.464, reg_lambda=0.8571, subsample=0.5213, random_state=7, nthread=-1)