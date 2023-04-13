import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
import os
import random
import random
from scipy.stats import norm
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input2 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
_input1.head()
_input1.columns
_input0.head()
_input2.head()
with open('_data/input/house-prices-advanced-regression-techniques/data_description.txt', encoding='utf8') as f:
    for line in f:
        print(line.strip())
_input1.describe()
_input1.shape
_input0.describe()
_input0.shape
_input1.info()
_input1.isnull().sum()[_input1.isnull().sum() != 0]
_input0.isnull().sum()[_input0.isnull().sum() != 0]
features = _input1.columns
total_houses = _input1.shape[0]
full = pd.DataFrame()
remove = pd.DataFrame()
medium = pd.DataFrame()
for feature in features:
    if _input1[feature].count() == total_houses:
        full[feature] = _input1[feature]
    elif _input1[feature].count() > 0.5 * total_houses:
        medium[feature] = _input1[feature]
    else:
        remove[feature] = _input1[feature]
remove
print('Number of Numerical feature: ', end=' ')
print(len(_input1.select_dtypes(include=['number']).columns))
print('Numerical features:')
print(_input1.select_dtypes(include=['number']).columns.values)
_input1.describe(exclude=['O'])
print('Number of Numerical feature: ', end=' ')
print(len(_input1.select_dtypes(include=['O']).columns))
print('Categorical  features:')
print(_input1.select_dtypes(include=['O']).columns.values)
plt.figure(figsize=(8, 8))
plt.hist(_input1['SalePrice'], bins=50)
plt.title('Sale Prices')
print(_input1['SalePrice'].describe())
print('Min Sale of the House:', 34900)
print('Max Sale of the House:', 755000)
_input1 = _input1.drop(['Id'], axis=1, inplace=False)
_input1 = _input1.drop(columns=remove, axis=1, inplace=False)
_input1
df_num = _input1.select_dtypes(include='number')
df_num
df_cat = _input1.select_dtypes(include='O')
df_cat
df_num.hist(figsize=(20, 25), bins=50, xlabelsize=8, ylabelsize=8)
_input1['PoolArea'].value_counts()
df_num.loc[df_num['GarageArea'] == 0, 'GarageArea']
feature_zero_ratio = {feature: df_num.loc[df_num[feature] == 0, feature].count() / 1460 for feature in df_num.columns.values}
feature_zero_ratio
for feature in df_num.columns.values:
    if feature_zero_ratio[feature] > 0.4:
        df_num = df_num.drop(columns=feature, axis=1, inplace=False)
        _input1 = _input1.drop([feature], axis=1)
        if feature in medium:
            medium = medium.drop([feature], axis=1)
print(_input1.shape)
print(df_num.shape)
df_num_corr = df_num.corr()['SalePrice'][:-1]
golden_features_list = df_num_corr[abs(df_num_corr) > 0.5].sort_values(ascending=False)
print('There is {} strongly correlated values with SalePrice:\n{}'.format(len(golden_features_list), golden_features_list))
threshold = 0.8

def highlight(value):
    if value > threshold:
        style = 'background-color: pink'
    else:
        style = 'background-color: palegreen'
    return style
corr_matrix = df_num.corr().abs().round(2)
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
upper.style.format('{:.2f}').applymap(highlight)
collinear_feature = [column for column in upper.columns if any(upper[column] > threshold)]
_input1 = _input1.drop(columns=collinear_feature, inplace=False)
df_num = df_num.drop(columns=collinear_feature, inplace=False)
for i in range(0, len(df_num.columns), 5):
    sns.pairplot(data=df_num, x_vars=df_num.columns[i:i + 5], y_vars=['SalePrice'])
print('Number of features left in numerical features:', len(df_num.columns))
print('Numerical Features left:')
print(df_num.columns.values)
df_num.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(df_num.corr(), annot=True, square=True, fmt='.2f', annot_kws={'size': 10})
corr_with_price = df_num.corr()
corr_with_price = corr_with_price.sort_values(by='SalePrice', ascending=False)
corr_with_price['SalePrice']
numerical_have_missing = pd.DataFrame()
categorical_have_missing = pd.DataFrame()
for feature in df_num.columns.values:
    if feature in medium:
        numerical_have_missing[feature] = df_num[feature]
for feature in df_cat.columns.values:
    if feature in medium:
        categorical_have_missing[feature] = df_cat[feature]
print(numerical_have_missing.columns)
df_num['LotFrontage'].describe()
df_num['LotFrontage'].hist()
old_LotFrontage = list(numerical_have_missing['LotFrontage'].values)
missing_idx = list(numerical_have_missing.loc[numerical_have_missing['LotFrontage'].isnull(), 'LotFrontage'].index)
rand_values = [random.randint(59, 80) for i in range(len(missing_idx))]
ind = 0
for idx in missing_idx:
    old_LotFrontage[idx] = rand_values[ind]
    ind += 1
numerical_have_missing['LotFrontage'] = pd.Series(old_LotFrontage)
_input1['LotFrontage'] = pd.Series(old_LotFrontage)
print(numerical_have_missing['LotFrontage'].count())
print(_input1['LotFrontage'].count())
print(categorical_have_missing.columns.values)
categorical_have_missing.isnull().sum()
print('Total categorical missing values:', len(categorical_have_missing.isnull().sum()))
(fig, axes) = plt.subplots(round(len(categorical_have_missing.columns) / 4), 4, figsize=(18, 10))
for (i, ax) in enumerate(fig.axes):
    if i < len(categorical_have_missing.columns):
        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)
        sns.countplot(x=categorical_have_missing.columns[i], alpha=0.7, data=categorical_have_missing, ax=ax)
fig.tight_layout()
categorical_have_missing = categorical_have_missing.drop(['FireplaceQu'], axis=1, inplace=False)
_input1 = _input1.drop(['FireplaceQu'], axis=1, inplace=False)
imputer = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')
for feature in categorical_have_missing.columns:
    categorical_have_missing[feature] = imputer.fit_transform(categorical_have_missing[feature].values.reshape(-1, 1))
    _input1[feature] = imputer.fit_transform(_input1[feature].values.reshape(-1, 1))
_input1.isnull().sum()
_input1.shape
for i in range(0, len(df_num.columns), 5):
    plt.figure(figsize=(15, 15))
    sns.pairplot(data=df_num, x_vars=df_num.columns[i:i + 5], y_vars=['SalePrice'])
outlier_indices = []
outlier_indices.extend(list(_input1[_input1['LotFrontage'] > 250].index))
plt.scatter(_input1['LotArea'], _input1['SalePrice'])
outlier_indices.extend(list(_input1[_input1['LotArea'] > 150000].index))
outlier_indices.extend(list(_input1[(_input1['BsmtFinSF1'] > 4000) | (_input1['TotalBsmtSF'] > 4000) | (_input1['GrLivArea'] > 4000) & (_input1['SalePrice'] < 400000)].index))
outlier_indices
_input1 = _input1.drop(_input1.index[outlier_indices], inplace=False)
_input1
_input1 = _input1.reset_index(drop=True, inplace=False)
sns.distplot(_input1['SalePrice'], fit=norm)
plt.figure()
res = stats.probplot(_input1['SalePrice'], plot=plt)
_input1['SalePrice'] = np.log(_input1['SalePrice'])
sns.distplot(_input1['SalePrice'], fit=norm)
plt.figure()
res = stats.probplot(_input1['SalePrice'], plot=plt)
sns.distplot(_input1['GrLivArea'], fit=norm)
fig = plt.figure()
res = stats.probplot(_input1['GrLivArea'], plot=plt)
_input1['GrLivArea'] = np.log(_input1['GrLivArea'])
sns.distplot(_input1['GrLivArea'], fit=norm)
fig = plt.figure()
res = stats.probplot(_input1['GrLivArea'], plot=plt)
from sklearn.preprocessing import LabelEncoder
df_cat = _input1.select_dtypes(include='O')
le = LabelEncoder()
for feature in df_cat.columns.values:
    df_cat[feature] = le.fit_transform(df_cat[feature])
    _input1[feature] = le.fit_transform(_input1[feature])
_input1.info()
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
target = _input1['SalePrice']
_input1 = _input1.drop(['SalePrice'], axis=1)
(X, y) = (_input1, target)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
lr = LinearRegression()