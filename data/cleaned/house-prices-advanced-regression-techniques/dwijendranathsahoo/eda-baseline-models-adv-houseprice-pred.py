import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
sample_sub = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
train.head()
train.shape
train.describe().T
train.info()
Missing_train = train.isnull().sum().sort_values(ascending=False)
Missing_train = Missing_train[Missing_train > 0]
percent = (train.isnull().sum() / train.isnull().count() * 100).sort_values(ascending=False)
percent = percent[percent > 0]
missing_data = pd.concat([Missing_train, percent], axis=1, keys=['Missing_train', 'Percent'])
(f, ax) = plt.subplots(figsize=(15, 6))
plt.xticks(rotation='90')
sns.barplot(x=missing_data.index, y=missing_data['Percent'])
plt.xlabel('features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
missing_data
train = train.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1)
features = [col for col in train.columns if col not in ['Id', 'SalePrice']]
print(features)
numerical_cols = [col for col in features if train[col].dtypes != 'O']
print(numerical_cols)
len(numerical_cols)
categorical_cols = [col for col in features if train[col].dtypes == 'O']
print(categorical_cols)
len(categorical_cols)
imputer = SimpleImputer(strategy='median')
train[numerical_cols] = imputer.fit_transform(train[numerical_cols])
scaler = StandardScaler()
train[numerical_cols] = scaler.fit_transform(train[numerical_cols])
print((train[categorical_cols].isnull().sum() / len(train) * 100).sort_values(ascending=False))
train_columns_None = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'GarageType', 'GarageFinish', 'GarageQual', 'FireplaceQu', 'GarageCond']
train[train_columns_None] = train[train_columns_None].fillna(train[train_columns_None].mode().iloc[0])
train_columns_None = ['MasVnrType', 'Electrical']
train[train_columns_None] = train[train_columns_None].fillna(train[train_columns_None].mode().iloc[0])
encoder = OrdinalEncoder()
train[categorical_cols] = encoder.fit_transform(train[categorical_cols])
test.head()
test.shape
test.describe().T
test.info()
test.describe().T
Missing_test = test.isnull().sum().sort_values(ascending=False)
Missing_test = Missing_test[Missing_test > 0]
percent = (test.isnull().sum() / test.isnull().count() * 100).sort_values(ascending=False)
percent = percent[percent > 0]
missing_data = pd.concat([Missing_test, percent], axis=1, keys=['Missing_test', 'Percent'])
(f, ax) = plt.subplots(figsize=(15, 6))
plt.xticks(rotation='90')
sns.barplot(x=missing_data.index, y=missing_data['Percent'])
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percentage of Missing Values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
missing_data
test = test.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1)
test[numerical_cols] = imputer.transform(test[numerical_cols])
test[numerical_cols] = scaler.transform(test[numerical_cols])
print((test[categorical_cols].isnull().sum() / len(test) * 100).sort_values(ascending=False))
test_columns_None = ['FireplaceQu', 'GarageCond', 'GarageQual', 'GarageFinish', 'GarageType', 'BsmtCond', 'BsmtExposure', 'BsmtQual', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType']
test[test_columns_None] = test[test_columns_None].fillna(test[test_columns_None].mode().iloc[0])
test_columns_None = ['MSZoning', 'Functional', 'Utilities', 'Exterior1st', 'Exterior2nd', 'SaleType', 'KitchenQual']
test[test_columns_None] = test[test_columns_None].fillna(test[test_columns_None].mode().iloc[0])
test[categorical_cols] = encoder.transform(test[categorical_cols])
test = test[numerical_cols + categorical_cols]
train.SalePrice.describe()
plt.figure(figsize=(10, 5))
sns.histplot(data=train, x=train['SalePrice'], kde=True)
print('Skewness: %f' % train['SalePrice'].skew())
print('Kurtosis: %f' % train['SalePrice'].kurt())
train['SalePrice'] = np.log1p(train['SalePrice'])
train.SalePrice.describe()
plt.figure(figsize=(10, 5))
sns.histplot(data=train, x=train['SalePrice'], kde=True)
print('Skewness: %f' % train['SalePrice'].skew())
print('Kurtosis: %f' % train['SalePrice'].kurt())
X = train[numerical_cols + categorical_cols]
y = train.SalePrice
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
fold = 0
for (train_idx, valid_idx) in kf.split(X, y):
    (X_train, X_valid) = (X.iloc[train_idx], X.iloc[valid_idx])
    (y_train, y_valid) = (y.iloc[train_idx], y.iloc[valid_idx])
    dtr = DecisionTreeRegressor()