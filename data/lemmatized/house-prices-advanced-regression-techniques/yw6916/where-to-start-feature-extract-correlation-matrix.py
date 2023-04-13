import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
_input1 = _input1.drop(['Id'], axis=1)
_input1.head()
_input1['SalePrice'].describe()
sns.distplot(_input1['SalePrice'])
data = pd.concat([_input1['SalePrice'], _input1['LotArea']], axis=1)
data.plot.scatter(x='LotArea', y='SalePrice', ylim=(0, 800000))
data = pd.concat([_input1['SalePrice'], _input1['GrLivArea']], axis=1)
data.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0, 800000))
_input1 = _input1.drop(_input1[(_input1['GrLivArea'] > 4000) & (_input1['SalePrice'] < 300000)].index)
data = pd.concat([_input1['SalePrice'], _input1['OverallQual']], axis=1)
fig = sns.boxplot(x='OverallQual', y='SalePrice', data=data)
fig.axis(ymin=0, ymax=800000)
pass
data = pd.concat([_input1['SalePrice'], _input1['Neighborhood']], axis=1)
(f, ax) = plt.subplots(figsize=(26, 12))
fig = sns.boxplot(x='Neighborhood', y='SalePrice', data=data)
fig.axis(ymin=0, ymax=800000)
data = pd.concat([_input1['SalePrice'], _input1['CentralAir']], axis=1)
(f, ax) = plt.subplots()
fig = sns.boxplot(x='CentralAir', y='SalePrice', data=data)
fig.axis(ymin=0, ymax=800000)
_input1['CentralAir'].replace(to_replace=['N', 'Y'], value=[0, 1])
_input1.head()
data = pd.concat([_input1['SalePrice'], _input1['GarageCars']], axis=1)
data.plot.scatter(x='GarageCars', y='SalePrice', ylim=(0, 800000))
data = pd.concat([_input1['SalePrice'], _input1['GarageArea']], axis=1)
data.plot.scatter(x='GarageArea', y='SalePrice', ylim=(0, 800000))
data = pd.concat([_input1['SalePrice'], _input1['YearBuilt']], axis=1)
data.plot.scatter(x='YearBuilt', y='SalePrice', ylim=(0, 800000))
(f, ax) = plt.subplots(figsize=(26, 12))
fig = sns.boxplot(x='YearBuilt', y='SalePrice', data=data)
fig.axis(ymin=0, ymax=800000)
from sklearn import preprocessing
f_names = ['CentralAir', 'Neighborhood']
for x in f_names:
    label = preprocessing.LabelEncoder()
    _input1[x] = label.fit_transform(_input1[x])
corrmat = _input1.corr()
(f, ax) = plt.subplots(figsize=(20, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)
k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(_input1[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
from sklearn import preprocessing
from sklearn import linear_model, svm, gaussian_process
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
cols = ['OverallQual', 'GrLivArea', 'GarageArea', '1stFlrSF', 'FullBath', 'YearBuilt']
x = _input1[cols].values
y = _input1['SalePrice'].values
x_scaled = preprocessing.StandardScaler().fit_transform(x)
y_scaled = preprocessing.StandardScaler().fit_transform(y.reshape(-1, 1))
(X_train, X_vali, y_train, y_vali) = train_test_split(x_scaled, y_scaled, test_size=0.3, random_state=42)
cols = ['OverallQual', 'GrLivArea', 'GarageArea', '1stFlrSF', 'FullBath', 'YearBuilt']
X_train = _input1[cols].values
y_train = _input1['SalePrice'].values
clf_1 = RandomForestRegressor(n_estimators=400)