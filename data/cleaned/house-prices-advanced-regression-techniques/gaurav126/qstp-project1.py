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
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
combine = pd.concat([train, test])
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)
train.head()
test.head()
(test.shape, train.shape)
train.describe().T
train['SalePrice']
correlation_num = train.corr()
correlation_num.sort_values(['SalePrice'], ascending=True, inplace=True)
correlation_num.SalePrice
plt.figure(figsize=(20, 10))
sns.heatmap(train.corr(), yticklabels=True, cbar=True, cmap='ocean')

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
details_tr = descr(train)
details_tr.reset_index(level=[0], inplace=True)
details_tr.sort_values(by='missing_percent', ascending=False)
details_tr.sort_values(by='missing_percent', ascending=False, inplace=True)
details_tr = details_tr[details_tr['missing_percent'] > 0]
plt.figure(figsize=(10, 4), dpi=100)
sns.barplot(x=details_tr['index'], y=details_tr['missing_percent'], data=details_tr)
plt.xticks(rotation=90)

details_test = descr(test)
details_test.reset_index(level=[0], inplace=True)
details_test.sort_values(by='missing_percent', ascending=False)
train.isnull().values.any()
test.isnull().values.any()
train['Electrical'].mode()
n = []
c = []
bsmt_str_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
bsmt_num_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']
for (col, col_df) in details_tr.iterrows():
    row = col_df['index']
    if col_df['dtypes'] == 'object':
        c.append(col)
        if row == 'Electrical':
            train[row].fillna('SBrkr', inplace=True)
        elif row == 'MasVnrType':
            train[row].fillna('None', inplace=True)
        elif row == 'GarageType':
            train[row].fillna('Attchd', inplace=True)
        elif row == 'GarageCond':
            train[row].fillna('TA', inplace=True)
        elif row == 'GarageFinish':
            train[row].fillna('Unf', inplace=True)
        elif row == 'GarageQual':
            train[row].fillna('TA', inplace=True)
        elif row == 'FireplaceQu':
            train[row].fillna('None', inplace=True)
        for i in bsmt_str_cols:
            if row == i:
                train[row].fillna('None', inplace=True)
        else:
            train[row].fillna('NotAvailable', inplace=True)
    else:
        n.append(col)
        if row == 'MasVnrArea':
            train[row].fillna(0, inplace=True)
        for i in bsmt_num_cols:
            if row == i:
                train[row].fillna('None', inplace=True)
        else:
            train[row].fillna(train[row].median(), inplace=True)
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
            test[row].fillna('SBrkr', inplace=True)
        elif row == 'MasVnrType':
            test[row].fillna('None', inplace=True)
        elif row == 'GarageType':
            test[row].fillna('Attchd', inplace=True)
        elif row == 'GarageCond':
            test[row].fillna('TA', inplace=True)
        elif row == 'GarageFinish':
            test[row].fillna('Unf', inplace=True)
        elif row == 'GarageQual':
            test[row].fillna('TA', inplace=True)
        elif row == 'FireplaceQu':
            test[row].fillna('None', inplace=True)
        else:
            test[row].fillna('NotAvailable', inplace=True)
        for i in bsmt_str_cols:
            if row == i:
                test[row].fillna('None', inplace=True)
    else:
        nt.append(col)
        if row == 'MasVnrArea':
            test[row].fillna(0, inplace=True)
        else:
            test[row].fillna(test[row].median(), inplace=True)
        for i in bsmt_num_cols:
            if row == i:
                test[row].fillna('None', inplace=True)
print('\nNumerical Features   -->', len(nt))
print('Categorical Features -->', len(ct))
details_tr = descr(train)
details_tr.sort_values(by='missing_percent', ascending=False).head()
train.isnull().values.any()
details_test = descr(test)
details_test.reset_index(level=[0], inplace=True)
details_test.sort_values(by='dtypes', ascending=True).head()
test.isnull().values.any()
train_num = train.select_dtypes(exclude='object')
train_cat = train.select_dtypes(include='object')
test_num = test.select_dtypes(exclude='object')
test_cat = test.select_dtypes(include='object')
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
train.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('Year Sold')
plt.ylabel('Median House Price')
plt.title('House Price vs YearSold')
train_map = train.copy()
test_map = test.copy()
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