import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from scipy.stats import skew
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
_input0.head()
_input1.corr()['SalePrice'].sort_values()
plt.figure(figsize=(12, 6))
(fig, axes) = plt.subplots(2, 3, figsize=(14, 12))
fig.subplots_adjust(wspace=0.5, hspace=0.5)
sns.scatterplot(x='OverallQual', y='SalePrice', data=_input1, ax=axes[0][0])
sns.scatterplot(x='GrLivArea', y='SalePrice', data=_input1, ax=axes[0][1])
sns.scatterplot(x='GarageCars', y='SalePrice', data=_input1, ax=axes[0][2])
sns.scatterplot(x='GarageArea', y='SalePrice', data=_input1, ax=axes[1][0])
sns.scatterplot(x='TotalBsmtSF', y='SalePrice', data=_input1, ax=axes[1][1])
sns.scatterplot(x='1stFlrSF', y='SalePrice', data=_input1, ax=axes[1][2])
outliers_indexes = _input1[(_input1['OverallQual'] > 8) & (_input1['SalePrice'] < 200000)].index
outliers_indexes = outliers_indexes.append(_input1[(_input1['GrLivArea'] > 4000) & (_input1['SalePrice'] < 300000)].index)
outliers_indexes = outliers_indexes.append(_input1[(_input1['GarageArea'] > 1200) & (_input1['SalePrice'] < 300000)].index)
outliers_indexes = outliers_indexes.append(_input1[(_input1['TotalBsmtSF'] > 6000) & (_input1['SalePrice'] < 200000)].index)
outliers_indexes = outliers_indexes.append(_input1[(_input1['TotalBsmtSF'] > 4000) & (_input1['SalePrice'] < 200000)].index)
print(outliers_indexes.unique())
_input1 = _input1.drop(outliers_indexes.unique(), axis=0)
all_data = pd.concat([_input1, _input0], axis=0)
all_data = all_data.iloc[:, :-1]
plt.figure(figsize=(12, 6))
(fig, axes) = plt.subplots(2, 3, figsize=(14, 12))
fig.subplots_adjust(wspace=0.5, hspace=0.5)
sns.scatterplot(x='OverallQual', y='SalePrice', data=_input1, ax=axes[0][0])
sns.scatterplot(x='GrLivArea', y='SalePrice', data=_input1, ax=axes[0][1])
sns.scatterplot(x='GarageCars', y='SalePrice', data=_input1, ax=axes[0][2])
sns.scatterplot(x='GarageArea', y='SalePrice', data=_input1, ax=axes[1][0])
sns.scatterplot(x='TotalBsmtSF', y='SalePrice', data=_input1, ax=axes[1][1])
sns.scatterplot(x='1stFlrSF', y='SalePrice', data=_input1, ax=axes[1][2])
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
plt.figure(figsize=(12, 6), dpi=200)
(fig, axes) = plt.subplots(2, 2, figsize=(14, 12))
fig.subplots_adjust(wspace=0.5, hspace=0.5)
sns.histplot(_input1['SalePrice'], kde=True, ax=axes[0][0])
axes[0][0].set_title('SalePrice')
sns.histplot(_input1['SalePrice'] ** 0.5, kde=True, ax=axes[0][1])
axes[0][1].set_title('Square Root Transformation')
sns.histplot(1 / _input1['SalePrice'], kde=True, ax=axes[1][0])
axes[1][0].set_title('Reciprocal Transformation')
sns.histplot(np.log1p(_input1['SalePrice']), kde=True, ax=axes[1][1])
axes[1][1].set_title('Log Transformation')
_input1['SalePrice'] = np.log1p(_input1['SalePrice'])
numeric_features = all_data.dtypes[all_data.dtypes != 'object'].index
skewed_features = _input1[numeric_features].apply(lambda x: skew(x.dropna()))
skewed_features = skewed_features[skewed_features > 0.75]
skewed_features = skewed_features.index
all_data[skewed_features] = np.log1p(all_data[skewed_features])
all_data.skew()
all_data.info()
percent_nan = all_data.isnull().sum() / len(all_data) * 100
percent_nan = percent_nan[percent_nan > 0].sort_values()
percent_nan
plt.figure(figsize=(12, 6), dpi=200)
sns.barplot(x=percent_nan.index, y=percent_nan)
plt.xticks(rotation=90)
df_numerical_features_zero = ['MasVnrArea', 'GarageYrBlt', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFinSF1', 'BsmtFinSF2', 'GarageCars', 'GarageArea', 'BsmtFullBath', 'BsmtHalfBath']
df_string_features_None = ['MasVnrType']
df_string_features_NA = ['BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtExposure', 'BsmtFinType2', 'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType', 'FireplaceQu', 'Fence', 'Alley', 'MiscFeature', 'PoolQC']
df_string_features_Other = ['Exterior1st', 'Exterior2nd']
df_string_features_Oth = ['SaleType']
all_data[df_numerical_features_zero] = all_data[df_numerical_features_zero].fillna(0)
all_data[df_string_features_None] = all_data[df_string_features_None].fillna('None')
all_data[df_string_features_NA] = all_data[df_string_features_NA].fillna('NA')
all_data[df_string_features_Other] = all_data[df_string_features_Other].fillna('Other')
all_data[df_string_features_Oth] = all_data[df_string_features_Oth].fillna('Oth')
plt.figure(figsize=(8, 12))
sns.boxplot(x='LotFrontage', y='Neighborhood', data=all_data, orient='h')
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda val: val.fillna(val.mean()))
all_data['Electrical'] = all_data.groupby(['Neighborhood'])['Electrical'].transform(lambda val: val.fillna(val.value_counts().index.tolist()[0]))
all_data['KitchenQual'] = all_data.groupby(['OverallQual'])['KitchenQual'].transform(lambda val: val.fillna(val.value_counts().index.tolist()[0]))
all_data['Utilities'] = all_data.groupby(['Neighborhood'])['Utilities'].transform(lambda val: val.fillna(val.value_counts().index.tolist()[0]))
all_data['Functional'] = all_data.groupby(['Neighborhood'])['Functional'].transform(lambda val: val.fillna(val.value_counts().index.tolist()[0]))
all_data['MSZoning'] = all_data.groupby(['Neighborhood'])['MSZoning'].transform(lambda val: val.fillna(val.value_counts().index.tolist()[0]))
all_data.info()
all_data = pd.get_dummies(all_data)
all_data
Final_train = all_data[:_input1.shape[0]]
Final_train = pd.concat([Final_train, _input1['SalePrice']], axis=1)
Final_test = all_data[_input1.shape[0]:]
X = Final_train.drop('SalePrice', axis=1)
y = Final_train['SalePrice']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.1, random_state=101)
param_grid = {'alpha': [0.004, 0.0045, 0.0035, 0.01], 'l1_ratio': [0.075, 0.065, 0.076, 0.077]}
base_elastic_model = ElasticNet(max_iter=1000000)
grid_model = GridSearchCV(estimator=base_elastic_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=2)