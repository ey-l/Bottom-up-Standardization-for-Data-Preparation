import numpy as np
import pandas as pd
import math
import seaborn as sns
import os
import xgboost
df_train_o = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df_test_o = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print(df_train_o.shape)
print(df_test_o.shape)
df_train_o.head()
df_train_o.info()

def getNumCatFeatures(df_train_o):
    numerical_feats = df_train_o.dtypes[df_train_o.dtypes != 'object'].index
    print('Number of Numerical features: ', len(numerical_feats))
    categorical_feats = df_train_o.dtypes[df_train_o.dtypes == 'object'].index
    print('Number of Categorical features: ', len(categorical_feats))
    return (numerical_feats, categorical_feats)
(numerical_feats, categorical_feats) = getNumCatFeatures(df_train_o)
print(df_train_o[numerical_feats].columns)
print('*' * 100)
print(df_train_o[categorical_feats].columns)
df_train_o['ExterCond'].head()
sns.distplot(df_train_o['SalePrice'])
df_train_o['SalePrice_Logged'] = np.log(df_train_o['SalePrice'])
sns.distplot(df_train_o['SalePrice_Logged'])
quantitative = [f for f in df_train_o.columns if df_train_o.dtypes[f] != 'object']
quantitative.remove('SalePrice')
quantitative.remove('Id')
quantitative.remove('SalePrice_Logged')
f = pd.melt(df_train_o, value_vars=quantitative)
g = sns.FacetGrid(f, col='variable', col_wrap=2, sharex=False, sharey=False)
g = g.map(sns.histplot, 'value')
listSkew = ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea']
for col in listSkew:
    df_train_o[col + '_Logged'] = np.log(df_train_o[col])
    df_train_o.drop(columns=[col])
for i in range(len(listSkew)):
    listSkew[i] = listSkew[i] + '_Logged'
quantitative = listSkew
f = pd.melt(df_train_o, value_vars=quantitative)
g = sns.FacetGrid(f, col='variable', col_wrap=2, sharex=False, sharey=False)
g = g.map(sns.histplot, 'value')
f = pd.melt(df_train_o, value_vars=['OverallQual', 'OverallCond'])
g = sns.FacetGrid(f, col='variable', col_wrap=2, sharex=False, sharey=False)
g = g.map(sns.histplot, 'value')
mul = df_train_o['OverallCond'] * df_train_o['OverallQual']
sns.histplot(mul)
mul2 = df_train_o['SalePrice'] * df_train_o['OverallCond']
mul2 = np.log(mul2)
sns.histplot(mul2)

def multiplier(df, features):
    for col in features:
        mul = df['SalePrice'] * df[col]
        mul = np.log(mul)
        df[col + '_Logged'] = mul
    f = pd.melt(df, value_vars=features)
    g = sns.FacetGrid(f, col='variable', col_wrap=2, sharex=False, sharey=False)
    g = g.map(sns.histplot, 'value')
    return df

def multiplier2(df, features):
    mul = df[features[0]] * df[features[1]]
    df['QualxCond'] = mul
    f = pd.melt(df, value_vars='QualxCond')
    g = sns.FacetGrid(f, col='variable', col_wrap=2, sharex=False, sharey=False)
    g = g.map(sns.histplot, 'value')
    return df
df_train_o = multiplier2(df_train_o, ['OverallCond', 'OverallQual'])
df_test_o = multiplier2(df_test_o, ['OverallCond', 'OverallQual'])

def checkINFNULL(X):
    for col in X.columns:
        if X[col].isnull().any():
            print('null')
            break
    print('no null')
    print(np.isinf(X).values.sum())
for x in listSkew:
    print(x)
    print(np.isinf(df_train_o[x]).values.sum())
for col in df_train_o.columns:
    nbr = df_train_o[col].isnull().sum()
    if nbr / 1460 > 0.5:
        print(f'{col}:{nbr / df_train_o.shape[0]}')
df_train = df_train_o.drop(columns=['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'Id'])
df_test = df_test_o.drop(columns=['Alley', 'PoolQC', 'Fence', 'MiscFeature'])
cols_with_missing = [col for col in df_train.columns if col != 'SalePrice' and df_train[col].isnull().any()]
cols_with_missing

def getColsWithMissingValue(df):
    cols_with_missing = [col for col in df.columns if col != 'SalePrice' and df[col].isnull().any()]
    cols_with_missing_num = []
    cols_with_missing_cat = []
    if len(cols_with_missing) == 0:
        print(f'There is no null/NA in this df')
        return (cols_with_missing_num, cols_with_missing_cat)
    for col in cols_with_missing:
        if df[col].dtypes != object:
            cols_with_missing_num.append(col)
            print(f'mean of {col}:{df[col].mean()}')
            print(f'median of {col}:{df[col].median()}')
            print(f'mode of {col}:{df[col].mode()}')
        else:
            cols_with_missing_cat.append(col)
            print(f'Value counts of {col}:\n{df[col].value_counts()}')
    return (cols_with_missing_num, cols_with_missing_cat)
(cols_with_missing_num, cols_with_missing_cat) = getColsWithMissingValue(df_train)
sns.distplot(df_train['MasVnrArea'])
df_train['CentralAir'].replace({'N': 0, 'Y': 1}, inplace=True)
df_test['CentralAir'].replace({'N': 0, 'Y': 1}, inplace=True)

def gradingconverter(df, listToBeConverted, quality):
    for i in listToBeConverted:
        df[i].replace(quality, inplace=True)
    return df
quality = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
quality_2 = {'Ex': 8, 'Gd': 6, 'TA': 4, 'Fa': 3, 'Po': 2}
x = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'GarageQual', 'GarageCond', 'FireplaceQu']
df_train = gradingconverter(df_train, x, quality_2)
df_test = gradingconverter(df_test, x, quality_2)
quality2 = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 0}
quality2_2 = {'Gd': 6, 'Av': 4, 'Mn': 2, 'No': 1}
x = ['BsmtExposure']
df_train = gradingconverter(df_train, x, quality2_2)
df_test = gradingconverter(df_test, x, quality2_2)
qua_na = ['BsmtQual', 'FireplaceQu', 'GarageQual', 'BsmtCond', 'GarageCond', 'BsmtExposure']

def pre_replaceNA_with_0(df, qua_na):
    for col in qua_na:
        df[col].fillna(value=0, inplace=True)
    return df
df_train = pre_replaceNA_with_0(df_train, qua_na)
df_test = pre_replaceNA_with_0(df_test, qua_na)
for col in cols_with_missing_num:
    print(f'mean of {col}:\n{df_train[col].mean()}')
    print(f'median of {col}:\n{df_train[col].median()}')
    print(f'mode of {col}:\n{df_train[col].mode()}')
(numerical_cols, categorical_cols) = getNumCatFeatures(df_train)
numerical_cols
categorical_cols
(num_cols_with_missing, cat_cols_with_missing) = getColsWithMissingValue(df_train)
num_cols_with_missing
(num_cols_with_missing_test, cat_cols_with_missing_test) = getColsWithMissingValue(df_test)
num_cols_with_missing_test

def replaceNA_with_Median(df, num_cols_with_missing):
    for col in num_cols_with_missing:
        median = df[col].median()
        df[col].fillna(value=median, inplace=True)
        df[col + 'was_missing'] = df[col].isnull().astype(int)
    return df
df_train = replaceNA_with_Median(df_train, num_cols_with_missing)
df_test = replaceNA_with_Median(df_test, num_cols_with_missing_test)
(num_cols_with_missing, cat_cols_with_missing) = getColsWithMissingValue(df_train)
num_cols_with_missing
(num_cols_with_missing_test, cat_cols_with_missing_test) = getColsWithMissingValue(df_test)
num_cols_with_missing_test
df_train.shape
'LotArea_Logged' in df_train.columns
df_test.shape
'LotArea_Logged' in df_test.columns
cat_cols_with_missing
for col in cat_cols_with_missing:
    print(f'{col}:\n{df_train[col].value_counts()}')
sumofnullrow = 0
for col in cat_cols_with_missing:
    nbr = df_train_o[col].isnull().sum()
    print(f'{col}:{nbr / df_train_o.shape[0]}')
    sumofnullrow += nbr
sumofnullrow

def replaceNA_with_0(df, cat_cols_with_missing):
    for col in cat_cols_with_missing:
        df[col].fillna(value='0', inplace=True)
        df[col + 'was_missing'] = df[col].isnull().astype(int)
    return df
df_train = replaceNA_with_0(df_train, cat_cols_with_missing)
df_test = replaceNA_with_0(df_test, cat_cols_with_missing_test)
(num_cols_with_missing, cat_cols_with_missing) = getColsWithMissingValue(df_train)
(num_cols_with_missing_test, cat_cols_with_missing_test) = getColsWithMissingValue(df_test)
(nums, cats) = getNumCatFeatures(df_train)
(nums_test, cats_test) = getNumCatFeatures(df_test)
cats
cats_test

def transformCatToNum_FrequencyEncoding(df, cats):
    for col in cats:
        x = df[col].value_counts()
        xdict = x.to_dict()
        df[col].replace(xdict, inplace=True)
    return df
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

def transformCatToNum_LabelEncoding(df, cats):
    for col in cats:
        df[col] = le.fit_transform(df[col])
    return df

def transformCatToNum_Factorizing(df, cats):
    for col in cats:
        (df[col], _) = df[col].factorize()
    return df

def transformCatToNum_OneHot(df, cats):
    return pd.get_dummies(df)
labelencoding = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'GarageQual', 'GarageCond', 'FireplaceQu', 'BsmtExposure']
df_train = transformCatToNum_FrequencyEncoding(df_train, cats)
df_test = transformCatToNum_FrequencyEncoding(df_test, cats_test)
(nums, cats) = getNumCatFeatures(df_train)
(nums_test, cats_test) = getNumCatFeatures(df_test)
np.setdiff1d(df_train.columns, df_test.columns)
np.setdiff1d(df_test.columns, df_train.columns)
(nums, cats) = getNumCatFeatures(df_train)
(nums_test, cats_test) = getNumCatFeatures(df_test)
df_train.head()
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt
from xgboost import XGBRegressor
from sklearn.feature_selection import mutual_info_regression, f_regression
df_train.filter(regex='Logged', axis=1).columns
X = df_train.loc[:, df_train.columns != 'SalePrice_Logged']
X = X.loc[:, X.columns != 'SalePrice']
y = df_train['SalePrice']
y_logged = df_train['SalePrice_Logged']
xgb_feat = XGBRegressor(n_estimators=100)