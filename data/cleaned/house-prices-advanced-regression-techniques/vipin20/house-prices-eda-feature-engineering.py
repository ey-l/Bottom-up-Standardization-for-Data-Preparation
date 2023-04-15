import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, train_test_split, KFold, cross_val_predict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

from sklearn.linear_model import LinearRegression, RidgeCV, Lasso, ElasticNetCV, BayesianRidge, LassoLarsIC
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.kernel_ridge import KernelRidge
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
import math
from sklearn.preprocessing import StandardScaler
import warnings as wr
wr.filterwarnings('ignore')
train_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train_data.head()
Sale_Price = train_data.iloc[:, 80]
Sale_Price.shape
train_data.shape
train = train_data.drop(['SalePrice'], axis=1)
train.head()
test.head()
test.shape
data = pd.concat([train, test], keys=['x', 'y'])
data = data.drop(['Id'], axis=1)
data.shape
plt.figure(figsize=(20, 6))
sns.heatmap(data.isnull(), yticklabels=False, cbar=True, cmap='mako')
total_null = data.isnull().sum().sort_values(ascending=False)
percentage = (data.isnull().sum() / data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total_null, percentage], axis=1, keys=['Total', 'Percentage'])
missing_data.head(20)
data = data.drop(missing_data[missing_data['Percentage'] > 0.05].index, 1)
data.isnull().sum()
num_col = data._get_numeric_data().columns.tolist()
num_col
cat_col = set(data.columns) - set(num_col)
cat_col
for col in num_col:
    data[col].fillna(data[col].mean(), inplace=True)
for col in cat_col:
    data[col].fillna(data[col].mode()[0], inplace=True)
for i in cat_col:
    print(data[i].value_counts())
df = data.drop(['RoofMatl', 'Heating', 'Condition2', 'BsmtCond', 'CentralAir', 'Functional', 'Electrical', 'LandSlope', 'ExterCond', 'Condition1', 'GarageArea', 'BsmtUnfSF', '3SsnPorch', 'MiscVal', 'BsmtFinType2', 'Utilities', 'Street', 'Exterior2nd', 'Neighborhood'], axis=1)
corrmat = train_data.corr()
top_corr_features = corrmat.index[abs(corrmat['SalePrice']) > 0.5]
plt.figure(figsize=(9, 9))
g = sns.heatmap(train_data[top_corr_features].corr(), annot=True, cmap='RdYlGn')
var = train_data[train_data.columns[1:]].corr()['SalePrice'][:]
var.sort_values(ascending=False)
df = df.drop(['MoSold', 'BsmtFinSF2', 'BsmtHalfBath', 'OverallCond', 'YrSold', 'MSSubClass', 'EnclosedPorch', 'KitchenAbvGr', 'ScreenPorch', '2ndFlrSF', 'OverallQual', 'GrLivArea'], axis=1)
df.shape
df.describe()
sns.distplot(train_data['SalePrice'])
print('Skewness coeff. is: %f' % train_data['SalePrice'].skew())
print('Kurtosis coeff. is: %f' % train_data['SalePrice'].kurt())
sns.kdeplot(data=train_data, x='SalePrice', hue='MoSold', fill=True, common_norm=False, palette='husl')
data_year_trend = pd.concat([train_data['SalePrice'], train_data['YearBuilt']], axis=1)
data_year_trend.plot.scatter(x='YearBuilt', y='SalePrice', ylim=(0, 800000))
data_bsmt_trend = pd.concat([train_data['SalePrice'], train_data['TotalBsmtSF']], axis=1)
data_bsmt_trend.plot.scatter(x='TotalBsmtSF', y='SalePrice', ylim=(0, 800000))
data_PoolArea_trend = pd.concat([train_data['SalePrice'], train_data['PoolArea']], axis=1)
data_PoolArea_trend.plot.scatter(x='PoolArea', y='SalePrice', ylim=(0, 800000))
data = pd.concat([train_data['SalePrice'], train_data['OverallQual']], axis=1)
(f, ax) = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='OverallQual', y='SalePrice', data=data)
fig.axis(ymin=0, ymax=800000)
corr_matrix = df.corr()
(f, ax1) = plt.subplots(figsize=(12, 9))
ax1 = sns.heatmap(corr_matrix, vmax=0.9)
df.shape
n_features = df.select_dtypes(exclude=['object']).columns
X = pd.get_dummies(df)
X.shape
scaler = StandardScaler()
X[X.columns] = scaler.fit_transform(X[X.columns])
Train_data = X.loc['x']
Train_data.shape
Test_data = X.loc['y']
Test_data.shape
Train_data.insert(2, column='SalePrice', value=Sale_Price)
Train_data.head()
x = Train_data.drop(['SalePrice'], axis=True)
y = Train_data['SalePrice']
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.25, random_state=40)
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=50, random_state=40, min_impurity_decrease=0.002, min_weight_fraction_leaf=0.001, min_samples_split=5)