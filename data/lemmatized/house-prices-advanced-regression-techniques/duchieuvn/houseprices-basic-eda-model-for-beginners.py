import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
df = pd.concat([_input1, _input0], ignore_index=True)
df.info()
print(_input1.shape)
print('Number of missing target:', _input1['SalePrice'].isnull().sum())
cols_fillna = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType', 'FireplaceQu', 'GarageQual', 'GarageCond', 'GarageFinish', 'GarageType', 'Electrical', 'KitchenQual', 'SaleType', 'Functional', 'Exterior2nd', 'Exterior1st', 'BsmtExposure', 'BsmtCond', 'BsmtQual', 'BsmtFinType1', 'BsmtFinType2', 'MSZoning', 'Utilities']
for col in cols_fillna:
    df[col] = df[col].fillna('None', inplace=False)
df_obj = df.select_dtypes(include='object')
df_num = df.select_dtypes(exclude='object')
num_features = df_num.columns
df_num.head()
obj_rate = pd.DataFrame({'count': df_obj.isnull().sum(), 'rate': df_obj.isnull().sum() * 100 / len(df_obj)})
obj_rate = obj_rate[obj_rate['rate'] > 0]
obj_rate
cols = df_obj.columns[1:]
n = len(cols)
n_col = 2
n_row = int(np.ceil(n / n_col))
fig = plt.figure(figsize=(n_col * 10, n_row * 6))
gs = fig.add_gridspec(n_row, n_col)
ax = [[fig.add_subplot(gs[i, j]) for j in range(n_col)] for i in range(n_row)]
for i in range(n_row):
    for j in range(n_col):
        sns.boxplot(x=cols[n_col * i + j], y='SalePrice', data=df, ax=ax[i][j])
drop_cols = ['Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Condition1', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'ExterCond', 'Foundation', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleCondition']
df = df.drop(columns=drop_cols, inplace=False)
df_obj = df.select_dtypes(include='object')
cols = df_obj.columns
n = len(cols)
n_col = 2
n_row = int(np.ceil(n / n_col))
fig = plt.figure(figsize=(n_col * 10, n_row * 6))
gs = fig.add_gridspec(n_row, n_col)
ax = [[fig.add_subplot(gs[i, j]) for j in range(n_col)] for i in range(n_row)]
for i in range(n_row):
    for j in range(n_col):
        sns.boxplot(x=cols[n_col * i + j], y='SalePrice', data=df, ax=ax[i][j])
cols = df_obj.columns
n_col = 2
n_row = int(np.ceil(len(cols) / n_col))
fig = plt.figure(figsize=(n_col * 10, n_row * 6))
gs = fig.add_gridspec(n_row, n_col)
ax = [[fig.add_subplot(gs[i, j]) for j in range(n_col)] for i in range(n_row)]
for i in range(n_row):
    for j in range(n_col):
        sns.countplot(x=cols[n_col * i + j], data=df, ax=ax[i][j])
cate_features = list(cols)
cate_features
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df[cols] = df[cols].apply(le.fit_transform)
df[cols].info()
n = len(cate_features)
n_col = 2
n_row = int(np.ceil(n / n_col))
fig = plt.figure(figsize=(n_col * 10, n_row * 6))
gs = fig.add_gridspec(n_row, n_col)
ax = [[fig.add_subplot(gs[i, j]) for j in range(n_col)] for i in range(n_row)]
for i in range(n_row):
    for j in range(n_col):
        sns.regplot(x=cate_features[n_col * i + j], y='SalePrice', data=df, ax=ax[i][j])
corr = df.corr()
abs_corr = corr['SalePrice'].abs()
print('print all absolute correlation')
print(abs_corr)
strong_corr = abs_corr[abs_corr > 0.4].index
print(strong_corr)
df = df[strong_corr]
df.info()
corr = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True)
cols = ['OverallQual', 'ExterQual', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'GarageYrBlt', 'GarageCars', 'GarageArea']
print(np.abs(corr['SalePrice'].loc[cols]))
df = df.drop(columns=['ExterQual', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'GarageYrBlt', 'GarageArea'])
corr = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True)
df_num = df.drop(columns=['SalePrice']).select_dtypes(exclude='object')
num_rate = pd.DataFrame({'count': df_num.isnull().sum(), 'rate': df_num.isnull().sum() * 100 / len(df_num)})
num_rate = num_rate[num_rate['rate'] > 0]
num_rate
df[num_rate.index] = df[num_rate.index].fillna(df[num_rate.index].mean().iloc[0])
df.info()
n = len(_input1)
df_train = df[:n]
df_test = df[n:]
print('Number of train sample:', len(df_train))
print('Number of test sample:', len(df_test))
num_features = _input1.select_dtypes(exclude='object').columns
cur_features = df.columns
num_features = list(set(num_features) & set(cur_features))
num_features
n = len(num_features)
fig = plt.figure(figsize=(6 * 2, 4 * n))
gs = fig.add_gridspec(n, 2)
ax = [[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(n)]
for i in range(n):
    sns.histplot(x=num_features[i], data=df_train, ax=ax[i][0])
    sns.boxplot(x=num_features[i], data=df_train, ax=ax[i][1])
df_train = df_train[df_train['TotalBsmtSF'] < 3000]
df_train = df_train[df_train['MasVnrArea'] < 900]
df_train = df_train[df_train['GrLivArea'] < 4000]
df_train.info()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_train_scaled = scaler.fit_transform(df_train)
df_test_scaled = scaler.transform(df_test)
y_train = df_train['SalePrice']
x_train = df_train.drop(columns=['SalePrice'])

def get_best_score(grid):
    best_score = np.sqrt(-grid.best_score_)
    print(best_score)
    print(grid.best_params_)
    print(grid.best_estimator_)
    return best_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
score_calc = 'neg_root_mean_squared_error'
nr_cv = 5
linreg = LinearRegression()
parameters = {'fit_intercept': [True, False], 'copy_X': [True, False]}
grid_linear = GridSearchCV(linreg, parameters, cv=nr_cv, verbose=1, scoring=score_calc)