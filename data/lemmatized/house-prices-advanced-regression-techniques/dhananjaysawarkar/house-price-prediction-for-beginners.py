import pandas as pd
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from scipy.stats import skew
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
(_input1.shape, _input0.shape)
_input1.columns
total = _input1.isnull().sum().sort_values(ascending=False)
percent = (_input1.isnull().sum() / _input1.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(30)
_input1 = _input1.drop(missing_data[missing_data['Total'] > 1].index, 1)
_input1['Electrical'] = _input1['Electrical'].fillna(_input1['Electrical'].mode()[0])
_input1.isnull().sum().max()
_input1.columns
total = _input0.isnull().sum().sort_values(ascending=False)
percent = (_input0.isnull().sum() / _input0.isnull().count()).sort_values(ascending=False)
missing_data1 = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data1.head(40)
_input0 = _input0.drop(missing_data1[missing_data1['Total'] > 4].index, 1)
total = _input0.isnull().sum().sort_values(ascending=False)
percent = (_input0.isnull().sum() / _input0.isnull().count()).sort_values(ascending=False)
missing_data1 = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data1.head(40)
null_features = missing_data1[missing_data1['Total'] > 0].index
null_features
for feature in null_features:
    _input0[feature] = _input0[feature].fillna(_input0[feature].mode()[0])
_input0.isnull().sum().max()
_input1['SalePrice'].describe()
sns.distplot(_input1['SalePrice'])
corrmat = _input1.corr()
(f, ax) = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)
corrmat = _input1.corr()
top_corr_features = corrmat.index[abs(corrmat['SalePrice']) > 0.5]
plt.figure(figsize=(10, 10))
sns.heatmap(_input1[top_corr_features].corr(), annot=True)
top_corr_features
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(_input1[cols], size=2.5)
categorical_features = _input1.select_dtypes(include=['object']).columns
numerical_features = _input1.select_dtypes(exclude=['object']).columns
train_num = _input1[numerical_features]
train_cat = _input1[categorical_features]
(train_cat.shape, train_num.shape)
skewness = train_num.apply(lambda x: skew(x))
skewness.sort_values(ascending=False)
skewness = skewness[abs(skewness) > 0.5]
skewness.index
train_cat.head()
train_cat = pd.get_dummies(train_cat)
train_cat.shape
train1 = pd.concat([train_cat, train_num], axis=1)
train1.shape
categorical_features = _input0.select_dtypes(include=['object']).columns
numerical_features = _input0.select_dtypes(exclude=['object']).columns
test_num = _input0[numerical_features]
test_cat = _input0[categorical_features]
(test_num.shape, test_cat.shape)
skewness = test_num.apply(lambda x: skew(x))
skewness.sort_values(ascending=False)
test_cat.head()
test_cat = pd.get_dummies(test_cat)
test_cat.head()
test1 = pd.concat([test_cat, test_num], axis=1)
test1.shape
min_threshold = train1.SalePrice.mean() - 3 * train1.SalePrice.std()
max_threshold = train1.SalePrice.mean() + 3 * train1.SalePrice.std()
(min_threshold, max_threshold)
train1 = train1[(train1.SalePrice < max_threshold) & train1.SalePrice > min_threshold]
(train1.shape, test1.shape)
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
cols = [col for col in train1.columns if col not in test1.columns]
cols.remove('SalePrice')
train1 = train1.drop(cols, axis=1)
X = train1.drop(['SalePrice'], axis=1)
y = train1['SalePrice']
(test1.shape, train1.shape)
train1.shape
from xgboost import XGBRegressor
(X_train, X_valid, y_train, y_valid) = train_test_split(X, y, random_state=0)
model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)