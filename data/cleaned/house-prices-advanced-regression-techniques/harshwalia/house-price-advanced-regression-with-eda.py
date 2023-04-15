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
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
sample = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
train.head()
train.columns
test.head()
sample.head()
with open('_data/input/house-prices-advanced-regression-techniques/data_description.txt', encoding='utf8') as f:
    for line in f:
        print(line.strip())
train.describe()
train.shape
test.describe()
test.shape
train.info()
train.isnull().sum()[train.isnull().sum() != 0]
test.isnull().sum()[test.isnull().sum() != 0]
features = train.columns
total_houses = train.shape[0]
full = pd.DataFrame()
remove = pd.DataFrame()
medium = pd.DataFrame()
for feature in features:
    if train[feature].count() == total_houses:
        full[feature] = train[feature]
    elif train[feature].count() > 0.5 * total_houses:
        medium[feature] = train[feature]
    else:
        remove[feature] = train[feature]
remove
print('Number of Numerical feature: ', end=' ')
print(len(train.select_dtypes(include=['number']).columns))
print('Numerical features:')
print(train.select_dtypes(include=['number']).columns.values)
train.describe(exclude=['O'])
print('Number of Numerical feature: ', end=' ')
print(len(train.select_dtypes(include=['O']).columns))
print('Categorical  features:')
print(train.select_dtypes(include=['O']).columns.values)
plt.figure(figsize=(8, 8))
plt.hist(train['SalePrice'], bins=50)
plt.title('Sale Prices')

print(train['SalePrice'].describe())
print('Min Sale of the House:', 34900)
print('Max Sale of the House:', 755000)
train.drop(['Id'], axis=1, inplace=True)
train.drop(columns=remove, axis=1, inplace=True)
train
df_num = train.select_dtypes(include='number')
df_num
df_cat = train.select_dtypes(include='O')
df_cat
df_num.hist(figsize=(20, 25), bins=50, xlabelsize=8, ylabelsize=8)
train['PoolArea'].value_counts()
df_num.loc[df_num['GarageArea'] == 0, 'GarageArea']
feature_zero_ratio = {feature: df_num.loc[df_num[feature] == 0, feature].count() / 1460 for feature in df_num.columns.values}
feature_zero_ratio
for feature in df_num.columns.values:
    if feature_zero_ratio[feature] > 0.4:
        df_num.drop(columns=feature, axis=1, inplace=True)
        train = train.drop([feature], axis=1)
        if feature in medium:
            medium = medium.drop([feature], axis=1)
print(train.shape)
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
train.drop(columns=collinear_feature, inplace=True)
df_num.drop(columns=collinear_feature, inplace=True)
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
train['LotFrontage'] = pd.Series(old_LotFrontage)
print(numerical_have_missing['LotFrontage'].count())
print(train['LotFrontage'].count())
print(categorical_have_missing.columns.values)
categorical_have_missing.isnull().sum()
print('Total categorical missing values:', len(categorical_have_missing.isnull().sum()))
(fig, axes) = plt.subplots(round(len(categorical_have_missing.columns) / 4), 4, figsize=(18, 10))
for (i, ax) in enumerate(fig.axes):
    if i < len(categorical_have_missing.columns):
        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)
        sns.countplot(x=categorical_have_missing.columns[i], alpha=0.7, data=categorical_have_missing, ax=ax)
fig.tight_layout()
categorical_have_missing.drop(['FireplaceQu'], axis=1, inplace=True)
train.drop(['FireplaceQu'], axis=1, inplace=True)
imputer = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')
for feature in categorical_have_missing.columns:
    categorical_have_missing[feature] = imputer.fit_transform(categorical_have_missing[feature].values.reshape(-1, 1))
    train[feature] = imputer.fit_transform(train[feature].values.reshape(-1, 1))
train.isnull().sum()
train.shape
for i in range(0, len(df_num.columns), 5):
    plt.figure(figsize=(15, 15))
    sns.pairplot(data=df_num, x_vars=df_num.columns[i:i + 5], y_vars=['SalePrice'])
outlier_indices = []
outlier_indices.extend(list(train[train['LotFrontage'] > 250].index))
plt.scatter(train['LotArea'], train['SalePrice'])
outlier_indices.extend(list(train[train['LotArea'] > 150000].index))
outlier_indices.extend(list(train[(train['BsmtFinSF1'] > 4000) | (train['TotalBsmtSF'] > 4000) | (train['GrLivArea'] > 4000) & (train['SalePrice'] < 400000)].index))
outlier_indices
train.drop(train.index[outlier_indices], inplace=True)
train
train.reset_index(drop=True, inplace=True)
sns.distplot(train['SalePrice'], fit=norm)
plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
train['SalePrice'] = np.log(train['SalePrice'])
sns.distplot(train['SalePrice'], fit=norm)
plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
sns.distplot(train['GrLivArea'], fit=norm)
fig = plt.figure()
res = stats.probplot(train['GrLivArea'], plot=plt)
train['GrLivArea'] = np.log(train['GrLivArea'])
sns.distplot(train['GrLivArea'], fit=norm)
fig = plt.figure()
res = stats.probplot(train['GrLivArea'], plot=plt)
from sklearn.preprocessing import LabelEncoder
df_cat = train.select_dtypes(include='O')
le = LabelEncoder()
for feature in df_cat.columns.values:
    df_cat[feature] = le.fit_transform(df_cat[feature])
    train[feature] = le.fit_transform(train[feature])
train.info()
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
target = train['SalePrice']
train = train.drop(['SalePrice'], axis=1)
(X, y) = (train, target)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
lr = LinearRegression()