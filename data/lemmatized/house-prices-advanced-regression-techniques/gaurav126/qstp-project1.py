import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_object_dtype
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from xgboost import XGBRegressor
from sklearn import preprocessing
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
combine = pd.concat([_input1, _input0])
_input1 = _input1.drop('Id', axis=1, inplace=False)
_input0 = _input0.drop('Id', axis=1, inplace=False)
_input1.head()
_input0.head()
(_input0.shape, _input1.shape)
_input1.describe().T
_input1['SalePrice']
correlation_num = _input1.corr()
correlation_num = correlation_num.sort_values(['SalePrice'], ascending=True, inplace=False)
correlation_num.SalePrice
plt.figure(figsize=(20, 10))
sns.heatmap(_input1.corr(), yticklabels=True, cbar=True, cmap='ocean')

def descr(train_num):
    no_rows = train_num.shape[0]
    types = train_num.dtypes
    col_null = train_num.columns[train_num.isna().any()].to_list()
    counts = train_num.apply(lambda x: x.count())
    uniques = train_num.apply(lambda x: x.unique())
    nulls = train_num.apply(lambda x: x.isnull().sum())
    distincts = train_num.apply(lambda x: x.unique().shape[0])
    nan_percent = train_num.isnull().sum() / no_rows * 100
    cols = {'dtypes': types, 'counts': counts, 'distincts': distincts, 'nulls': nulls, 'missing_percent': nan_percent, 'uniques': uniques}
    table = pd.DataFrame(data=cols)
    return table
details_tr = descr(_input1)
details_tr = details_tr.reset_index(level=[0], inplace=False)
details_tr.sort_values(by='missing_percent', ascending=False)
details_tr = details_tr.sort_values(by='missing_percent', ascending=False, inplace=False)
details_tr = details_tr[details_tr['missing_percent'] > 0]
plt.figure(figsize=(10, 4), dpi=100)
sns.barplot(x=details_tr['index'], y=details_tr['missing_percent'], data=details_tr)
plt.xticks(rotation=90)
details_test = descr(_input0)
details_test = details_test.reset_index(level=[0], inplace=False)
details_test.sort_values(by='missing_percent', ascending=False)
_input1.isnull().values.any()
_input0.isnull().values.any()
_input1['Electrical'].mode()
n = []
c = []
bsmt_str_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
bsmt_num_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']
for (col, col_df) in details_tr.iterrows():
    row = col_df['index']
    if col_df['dtypes'] == 'object':
        c.append(col)
        if row == 'Electrical':
            _input1[row] = _input1[row].fillna('SBrkr', inplace=False)
        elif row == 'MasVnrType':
            _input1[row] = _input1[row].fillna('None', inplace=False)
        elif row == 'GarageType':
            _input1[row] = _input1[row].fillna('Attchd', inplace=False)
        elif row == 'GarageCond':
            _input1[row] = _input1[row].fillna('TA', inplace=False)
        elif row == 'GarageFinish':
            _input1[row] = _input1[row].fillna('Unf', inplace=False)
        elif row == 'GarageQual':
            _input1[row] = _input1[row].fillna('TA', inplace=False)
        elif row == 'FireplaceQu':
            _input1[row] = _input1[row].fillna('None', inplace=False)
        for i in bsmt_str_cols:
            if row == i:
                _input1[row] = _input1[row].fillna('None', inplace=False)
        else:
            _input1[row] = _input1[row].fillna('NotAvailable', inplace=False)
    else:
        n.append(col)
        if row == 'MasVnrArea':
            _input1[row] = _input1[row].fillna(0, inplace=False)
        for i in bsmt_num_cols:
            if row == i:
                _input1[row] = _input1[row].fillna('None', inplace=False)
        else:
            _input1[row] = _input1[row].fillna(_input1[row].median(), inplace=False)
print('\nNumerical Features   -->', len(n))
print('Categorical Features -->', len(c))
nt = []
ct = []
bsmt_str_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
bsmt_num_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']
for (col, col_df) in details_test.iterrows():
    row = col_df['index']
    if col_df['dtypes'] == 'object':
        ct.append(col)
        if row == 'Electrical':
            _input0[row] = _input0[row].fillna('SBrkr', inplace=False)
        elif row == 'MasVnrType':
            _input0[row] = _input0[row].fillna('None', inplace=False)
        elif row == 'GarageType':
            _input0[row] = _input0[row].fillna('Attchd', inplace=False)
        elif row == 'GarageCond':
            _input0[row] = _input0[row].fillna('TA', inplace=False)
        elif row == 'GarageFinish':
            _input0[row] = _input0[row].fillna('Unf', inplace=False)
        elif row == 'GarageQual':
            _input0[row] = _input0[row].fillna('TA', inplace=False)
        elif row == 'FireplaceQu':
            _input0[row] = _input0[row].fillna('None', inplace=False)
        else:
            _input0[row] = _input0[row].fillna('NotAvailable', inplace=False)
        for i in bsmt_str_cols:
            if row == i:
                _input0[row] = _input0[row].fillna('None', inplace=False)
    else:
        nt.append(col)
        if row == 'MasVnrArea':
            _input0[row] = _input0[row].fillna(0, inplace=False)
        else:
            _input0[row] = _input0[row].fillna(_input0[row].median(), inplace=False)
        for i in bsmt_num_cols:
            if row == i:
                _input0[row] = _input0[row].fillna('None', inplace=False)
print('\nNumerical Features   -->', len(nt))
print('Categorical Features -->', len(ct))
details_tr = descr(_input1)
details_tr.sort_values(by='missing_percent', ascending=False).head()
_input1.isnull().values.any()
details_test = descr(_input0)
details_test = details_test.reset_index(level=[0], inplace=False)
details_test.sort_values(by='dtypes', ascending=True).head()
_input0.isnull().values.any()
train_num = _input1.select_dtypes(exclude='object')
train_cat = _input1.select_dtypes(include='object')
test_num = _input0.select_dtypes(exclude='object')
test_cat = _input0.select_dtypes(include='object')
for i in train_num.columns:
    sns.set_style('whitegrid')
    plt.figure(figsize=(10, 10))
    x = train_num[i]
    sns.jointplot(x=x, y=train_num['SalePrice'], data=train_num)
for i in train_cat.columns:
    sns.set_style('whitegrid')
    plt.figure(figsize=(15, 15))
    x = train_cat[i]
    sns.jointplot(x=x, y=train_num['SalePrice'], data=train_cat)
_input1.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('Year Sold')
plt.ylabel('Median House Price')
plt.title('House Price vs YearSold')
train_map = _input1.copy()
test_map = _input0.copy()
train_map.head()
for feature in train_map.select_dtypes(include='object'):
    labels_ordered = train_map.groupby([feature])['SalePrice'].mean().sort_values().index
    labels_ordered = {k: i for (i, k) in enumerate(labels_ordered, 0)}
    train_map[feature] = train_map[feature].map(labels_ordered)
for feature in test_map.select_dtypes(include='object'):
    labels_ordered = test_map.groupby([feature])['LotFrontage'].mean().sort_values().index
    labels_ordered = {k: i for (i, k) in enumerate(labels_ordered, 0)}
    test_map[feature] = test_map[feature].map(labels_ordered)
test_map.head()
train_map.head()
test_map = test_map.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1)
train_map = train_map.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1)
X = train_map.drop(['SalePrice'], axis=1).drop(train_map.index[-1])
Y = train_map['SalePrice'].drop(train_map.index[-1])
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.4, random_state=101)
scaler = StandardScaler()