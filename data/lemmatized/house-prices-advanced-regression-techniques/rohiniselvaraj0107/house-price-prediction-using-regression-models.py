import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
_input0.head()
_input1.describe()
_input0.describe()
_input1.shape
_input0.shape
train_missing = []
for col in _input1.columns:
    if _input1[col].isna().sum() != 0:
        missing = _input1[col].isna().sum()
        print(f'{col: <{20}} {missing}')
        if missing > _input1.shape[0] / 3:
            train_missing.append(col)
test_missing = []
for col in _input0.columns:
    if _input0[col].isna().sum() != 0:
        missing = _input0[col].isna().sum()
        print(f'{col: <{20}} {missing}')
        if missing > _input0.shape[0] / 3:
            test_missing.append(col)
cols = _input1.columns
no_of_cols = len(cols)
print('Number of columns: {}'.format(no_of_cols))
numeric_cols = [col for col in _input1.columns if _input1[col].dtype in ('int32', 'int64', 'float64')]
print('Numeric columns: {}'.format(numeric_cols))
print('No of numeric columns: {}'.format(len(numeric_cols)))
categorical_cols = [col for col in _input1.columns if _input1[col].dtype == 'object']
print('Categorical columns: {}'.format(categorical_cols))
print('No of categorical columns: {}'.format(len(categorical_cols)))
cols = _input0.columns
no_of_cols = len(cols)
print('Number of columns: {}'.format(no_of_cols))
numeric_cols = [col for col in _input0.columns if _input0[col].dtype in ('int32', 'int64', 'float64')]
print('Numeric columns: {}'.format(numeric_cols))
print('No of numeric columns: {}'.format(len(numeric_cols)))
categorical_cols = [col for col in _input0.columns if _input0[col].dtype == 'object']
print('Categorical columns: {}'.format(categorical_cols))
print('No of categorical columns: {}'.format(len(categorical_cols)))
plt.figure(figsize=(14, 12))
sns.heatmap(_input1.corr(), vmax=0.8, square=True, cmap='Blues')
corr_cols = _input1.corr().nlargest(15, 'SalePrice')['SalePrice'].index
plt.figure(figsize=(12, 10))
sns.heatmap(_input1[corr_cols].corr(), annot=True, vmax=0.8, square=True, cmap='Blues')
for col in _input1[categorical_cols]:
    if _input1[col].isnull().sum() > 0:
        print(col, _input1[col].unique())
col_fillna = ('Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature')
for col in col_fillna:
    _input1[col] = _input1[col].fillna('no', inplace=False)
    _input0[col] = _input0[col].fillna('no', inplace=False)
total = _input1.isnull().sum().sort_values(ascending=False)
percent = (_input1.isnull().sum() / _input1.isnull().count()).sort_values(ascending=False)
total_missing = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
total_missing
total = _input0.isnull().sum().sort_values(ascending=False)
percent = (_input0.isnull().sum() / _input0.isnull().count()).sort_values(ascending=False)
total_missing = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
total_missing
_input1 = _input1.fillna(_input1.median(), inplace=False)
_input0 = _input0.fillna(_input0.median(), inplace=False)
print(_input1.isnull().sum().sum())
print(_input0.isnull().sum().sum())
_input0.isnull().sum().sort_values(ascending=False)[:8]
for col in _input0.columns:
    if _input0[col].dtype == 'object' and _input0[col].isnull().sum():
        _input0 = _input0.fillna('no', inplace=False)
_input0.isnull().sum().sum()
from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder()
_input1[categorical_cols] = oe.fit_transform(_input1[categorical_cols])
_input0[categorical_cols] = oe.fit_transform(_input0[categorical_cols])
x = _input1.drop('SalePrice', axis=1)
y = _input1.SalePrice.copy()
from sklearn.feature_selection import mutual_info_classif
important = mutual_info_classif(x, y)
important_features = pd.Series(important, _input1.columns[:len(_input1.columns) - 1])
important_features
feature_cols = []
for (key, val) in important_features.items():
    if np.abs(val) > 0.5:
        feature_cols.append(key)
print(feature_cols)
print(len(feature_cols))
x = x[feature_cols]
xtest = _input0[feature_cols]
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2, random_state=0)
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')