import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import gc
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print('train.csv. Shape: ', _input1.shape)
print('test.csv. Shape: ', _input0.shape)
_input1['SalePrice'].describe()
(f, ax) = plt.subplots(figsize=(8, 6))
sns.distplot(_input1['SalePrice'])
print('Skewness: %f' % _input1['SalePrice'].skew())
print('Kurtosis: %f' % _input1['SalePrice'].kurt())
fig = plt.figure(figsize=(15, 10))
fig.add_subplot(1, 2, 1)
res = stats.probplot(_input1['SalePrice'], plot=plt)
fig.add_subplot(1, 2, 2)
res = stats.probplot(np.log1p(_input1['SalePrice']), plot=plt)
_input1['SalePrice'] = np.log1p(_input1['SalePrice'])
_input1['SalePrice'].head()
k = 15
corrmat = abs(_input1.corr(method='spearman'))
cols = corrmat.nlargest(k, 'SalePrice').index
cm = np.corrcoef(_input1[cols].values.T)
sns.set(font_scale=1.25)
(f, ax) = plt.subplots(figsize=(8, 6))
mask = np.zeros_like(cm)
mask[np.triu_indices_from(mask)] = True
sns.set_style('white')
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, yticklabels=cols.values, xticklabels=cols.values, mask=mask)
data = pd.concat([_input1['SalePrice'], _input1['OverallQual']], axis=1)
(f, ax) = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='OverallQual', y='SalePrice', data=data)
data = pd.concat([_input1['SalePrice'], _input1['GrLivArea']], axis=1)
(f, ax) = plt.subplots(figsize=(8, 6))
fig = sns.regplot(x='GrLivArea', y='SalePrice', data=data)
data = pd.concat([_input1['SalePrice'], _input1['GarageCars']], axis=1)
(f, ax) = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='GarageCars', y='SalePrice', data=data)
data = pd.concat([_input1['SalePrice'], _input1['YearBuilt']], axis=1)
(f, ax) = plt.subplots(figsize=(8, 6))
fig = sns.regplot(x='YearBuilt', y='SalePrice', data=data)
data = pd.concat([_input1['SalePrice'], _input1['GarageArea']], axis=1)
(f, ax) = plt.subplots(figsize=(8, 6))
fig = sns.regplot(x='GarageArea', y='SalePrice', data=data)
data = pd.concat([_input1['SalePrice'], _input1['TotalBsmtSF']], axis=1)
(f, ax) = plt.subplots(figsize=(8, 6))
fig = sns.regplot(x='TotalBsmtSF', y='SalePrice', data=data)
data = pd.concat([_input1['SalePrice'], _input1['FullBath']], axis=1)
(f, ax) = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='FullBath', y='SalePrice', data=data)
data = pd.concat([_input1['SalePrice'], _input1['TotRmsAbvGrd']], axis=1)
(f, ax) = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='TotRmsAbvGrd', y='SalePrice', data=data)
data = pd.concat([_input1['SalePrice'], _input1['1stFlrSF']], axis=1)
(f, ax) = plt.subplots(figsize=(8, 6))
fig = sns.regplot(x='1stFlrSF', y='SalePrice', data=data)
data = pd.concat([_input1['SalePrice'], _input1['LotArea']], axis=1)
(f, ax) = plt.subplots(figsize=(8, 6))
fig = sns.regplot(x='LotArea', y='SalePrice', data=data)
categorical_features = _input1.select_dtypes(include=['object']).columns
ix = 1
fig = plt.figure(figsize=(15, 10))
for c in list(_input1[categorical_features]):
    if ix <= 3:
        ax2 = fig.add_subplot(2, 3, ix + 3)
        sns.boxplot(data=_input1, x=c, y='SalePrice', ax=ax2)
    ix = ix + 1
    if ix == 4:
        fig = plt.figure(figsize=(15, 10))
        ix = 1
total = _input1.isnull().sum().sort_values(ascending=False)
percent = (_input1.isnull().sum() / _input1.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
percent_data = percent.head(20)
percent_data.plot(kind='bar', figsize=(8, 6), fontsize=10)
plt.xlabel('Columns', fontsize=20)
plt.ylabel('Count', fontsize=20)
plt.title('Total Missing Value (%)', fontsize=20)
Numeric = _input1.copy()
del Numeric['SalePrice']
Numeric_columns = Numeric.select_dtypes(include=['int64', 'float64']).columns
ix = 1
fig = plt.figure(figsize=(15, 10))
for c in list(Numeric_columns):
    if ix <= 3:
        ax2 = fig.add_subplot(2, 3, ix + 3)
        sns.distplot(_input1[c].dropna())
        sns.distplot(_input0[c].dropna())
        plt.legend(['train', 'test'])
        plt.grid()
    ix = ix + 1
    if ix == 4:
        fig = plt.figure(figsize=(15, 10))
        ix = 1
del Numeric
_input1 = _input1[_input1['Id'] != 692][_input1['Id'] != 1183]
_input1 = _input1.drop(_input1[(_input1['GrLivArea'] > 4000) & (_input1['SalePrice'] < 12.5)].index)
_input1 = _input1.drop(_input1[_input1['LotArea'] > 150000].index)
_input1 = _input1.drop(_input1[(_input1['GarageArea'] > 1200) & (_input1['SalePrice'] < 12.5)].index)
total = _input1.isnull().sum().sort_values(ascending=False)
percent = (_input1.isnull().sum() / _input1.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
percent_data = percent.head(20)
percent_data.plot(kind='bar', figsize=(8, 6), fontsize=10)
plt.xlabel('Columns', fontsize=20)
plt.ylabel('Count', fontsize=20)
plt.title('Total Missing Value (%)', fontsize=20)
import missingno as msno
len_train = _input1.shape[0]
y_reg = _input1['SalePrice']
Id = _input0['Id']
df_all = pd.concat([_input1, _input0])
del df_all['Id']
missingdata_df = df_all.columns[df_all.isnull().any()].tolist()
msno.heatmap(df_all[missingdata_df], figsize=(20, 20))
df_all['Utilities'].unique()
df_all['Utilities'].describe()
del df_all['Utilities']
df_all['PoolQC'] = df_all['PoolQC'].fillna('None')
df_all['MiscFeature'] = df_all['MiscFeature'].fillna('None')
df_all['Alley'] = df_all['Alley'].fillna('None')
df_all['Fence'] = df_all['Fence'].fillna('None')
df_all['FireplaceQu'] = df_all['FireplaceQu'].fillna('None')
df_all['BsmtQual'] = df_all['BsmtQual'].fillna('None')
df_all['BsmtCond'] = df_all['BsmtCond'].fillna('None')
df_all['BsmtExposure'] = df_all['BsmtExposure'].fillna('None')
df_all['BsmtFinType1'] = df_all['BsmtFinType1'].fillna('None')
df_all['BsmtFinType2'] = df_all['BsmtFinType2'].fillna('None')
df_all['GarageType'] = df_all['GarageType'].fillna('None')
df_all['GarageFinish'] = df_all['GarageFinish'].fillna('None')
df_all['GarageQual'] = df_all['GarageQual'].fillna('None')
df_all['GarageCond'] = df_all['GarageCond'].fillna('None')
df_all['BsmtFinSF1'] = df_all['BsmtFinSF1'].fillna(0)
df_all['BsmtFinSF2'] = df_all['BsmtFinSF2'].fillna(0)
df_all['BsmtUnfSF'] = df_all['BsmtUnfSF'].fillna(0)
df_all['TotalBsmtSF'] = df_all['TotalBsmtSF'].fillna(0)
df_all['BsmtFullBath'] = df_all['BsmtFullBath'].fillna(0)
df_all['BsmtHalfBath'] = df_all['BsmtHalfBath'].fillna(0)
df_all['MasVnrArea'] = df_all['MasVnrArea'].fillna(0)
df_all['GarageYrBlt'] = df_all['GarageYrBlt'].fillna(0)
df_all['GarageCars'] = df_all['GarageCars'].fillna(0)
df_all['GarageArea'] = df_all['GarageArea'].fillna(0)
df_all['MSZoning'] = df_all['MSZoning'].fillna(df_all['MSZoning'].mode()[0])
df_all['Exterior1st'] = df_all['Exterior1st'].fillna(df_all['Exterior1st'].mode()[0])
df_all['Exterior2nd'] = df_all['Exterior2nd'].fillna(df_all['Exterior2nd'].mode()[0])
df_all['MasVnrType'] = df_all['MasVnrType'].fillna(df_all['MasVnrType'].mode()[0])
df_all['Electrical'] = df_all['Electrical'].fillna(df_all['Electrical'].mode()[0])
df_all['KitchenQual'] = df_all['KitchenQual'].fillna(df_all['KitchenQual'].mode()[0])
df_all['Functional'] = df_all['Functional'].fillna(df_all['Functional'].mode()[0])
df_all['SaleType'] = df_all['SaleType'].fillna(df_all['SaleType'].mode()[0])
df_all[df_all['Neighborhood'] == 'BrkSide']['LotFrontage'].describe()
df_all[df_all['Neighborhood'] == 'CollgCr']['LotFrontage'].describe()
df_all['LotFrontage'] = df_all.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
total = df_all.isnull().sum().sort_values(ascending=False)
percent = (df_all.isnull().sum() / df_all.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
percent_data = percent.head(20)
percent_data.plot(kind='bar', figsize=(8, 6), fontsize=10)
plt.xlabel('Columns', fontsize=20)
plt.ylabel('Count', fontsize=20)
plt.title('Total Missing Value (%)', fontsize=20)
df_all['MSSubClass'] = df_all['MSSubClass'].apply(str)
df_all['OverallCond'] = df_all['OverallCond'].astype(str)
categorical_features = df_all.select_dtypes(include=['object']).columns
numerical_features = df_all.select_dtypes(exclude=['object']).columns
numerical_features = numerical_features.drop('SalePrice')
print('Numerical features : ' + str(len(numerical_features)))
print('Categorical features : ' + str(len(categorical_features)))
from sklearn.preprocessing import OneHotEncoder
one_hot_encoding = df_all.copy()
one_hot_encoding = pd.get_dummies(one_hot_encoding)
one_hot_encoding.iloc[:, 36:50].head()
del one_hot_encoding
label_encoding = df_all.copy()
for i in categorical_features:
    (label_encoding[i], indexer) = pd.factorize(label_encoding[i])
label_encoding.iloc[:, 20:30].head()
del label_encoding
frequency_encoding_all = df_all.copy()

def frequency_encoding(frame, col):
    freq_encoding = frame.groupby([col]).size() / frame.shape[0]
    freq_encoding = freq_encoding.reset_index().rename(columns={0: '{}_Frequency'.format(col)})
    return frame.merge(freq_encoding, on=col, how='left')
for col in categorical_features:
    frequency_encoding_all = frequency_encoding(frequency_encoding_all, col)
frequency_encoding_all = frequency_encoding_all.drop(categorical_features, axis=1, inplace=False)
frequency_encoding_all.iloc[:, 20:30].head()
del frequency_encoding_all
from sklearn.model_selection import KFold, cross_val_score, train_test_split
n_folds = 5

def rmsle_cv(model, df):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(df.values)
    rmse = np.sqrt(-cross_val_score(model, df.values, y_reg, scoring='neg_mean_squared_error', cv=kf))
    return rmse
one_hot_encoding = df_all.copy()
del one_hot_encoding['SalePrice']
one_hot_encoding = pd.get_dummies(one_hot_encoding)
one_hot_encoding_train = one_hot_encoding[:len_train]
one_hot_encoding_test = one_hot_encoding[len_train:]
del one_hot_encoding
import lightgbm as lgb
model = lgb.LGBMRegressor(objective='regression', num_leaves=5, learning_rate=0.05, n_estimators=720, max_bin=55, bagging_fraction=0.8, bagging_freq=5, feature_fraction=0.2319, feature_fraction_seed=9, bagging_seed=9, min_data_in_leaf=6, min_sum_hessian_in_leaf=11)
score = rmsle_cv(model, one_hot_encoding_train)
print('One-hot encoding(5-folds) LGBM score: {:.4f} ({:.4f})\n'.format(score.mean(), score.std()))

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))