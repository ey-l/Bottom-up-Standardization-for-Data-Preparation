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
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input2 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
_input1.head()
_input1.shape
_input1.describe().T
_input1.info()
Missing_train = _input1.isnull().sum().sort_values(ascending=False)
Missing_train = Missing_train[Missing_train > 0]
percent = (_input1.isnull().sum() / _input1.isnull().count() * 100).sort_values(ascending=False)
percent = percent[percent > 0]
missing_data = pd.concat([Missing_train, percent], axis=1, keys=['Missing_train', 'Percent'])
(f, ax) = plt.subplots(figsize=(15, 6))
plt.xticks(rotation='90')
sns.barplot(x=missing_data.index, y=missing_data['Percent'])
plt.xlabel('features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
missing_data
_input1 = _input1.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1)
features = [col for col in _input1.columns if col not in ['Id', 'SalePrice']]
print(features)
numerical_cols = [col for col in features if _input1[col].dtypes != 'O']
print(numerical_cols)
len(numerical_cols)
categorical_cols = [col for col in features if _input1[col].dtypes == 'O']
print(categorical_cols)
len(categorical_cols)
imputer = SimpleImputer(strategy='median')
_input1[numerical_cols] = imputer.fit_transform(_input1[numerical_cols])
scaler = StandardScaler()
_input1[numerical_cols] = scaler.fit_transform(_input1[numerical_cols])
print((_input1[categorical_cols].isnull().sum() / len(_input1) * 100).sort_values(ascending=False))
train_columns_None = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'GarageType', 'GarageFinish', 'GarageQual', 'FireplaceQu', 'GarageCond']
_input1[train_columns_None] = _input1[train_columns_None].fillna(_input1[train_columns_None].mode().iloc[0])
train_columns_None = ['MasVnrType', 'Electrical']
_input1[train_columns_None] = _input1[train_columns_None].fillna(_input1[train_columns_None].mode().iloc[0])
encoder = OrdinalEncoder()
_input1[categorical_cols] = encoder.fit_transform(_input1[categorical_cols])
_input0.head()
_input0.shape
_input0.describe().T
_input0.info()
_input0.describe().T
Missing_test = _input0.isnull().sum().sort_values(ascending=False)
Missing_test = Missing_test[Missing_test > 0]
percent = (_input0.isnull().sum() / _input0.isnull().count() * 100).sort_values(ascending=False)
percent = percent[percent > 0]
missing_data = pd.concat([Missing_test, percent], axis=1, keys=['Missing_test', 'Percent'])
(f, ax) = plt.subplots(figsize=(15, 6))
plt.xticks(rotation='90')
sns.barplot(x=missing_data.index, y=missing_data['Percent'])
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percentage of Missing Values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
missing_data
_input0 = _input0.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1)
_input0[numerical_cols] = imputer.transform(_input0[numerical_cols])
_input0[numerical_cols] = scaler.transform(_input0[numerical_cols])
print((_input0[categorical_cols].isnull().sum() / len(_input0) * 100).sort_values(ascending=False))
test_columns_None = ['FireplaceQu', 'GarageCond', 'GarageQual', 'GarageFinish', 'GarageType', 'BsmtCond', 'BsmtExposure', 'BsmtQual', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType']
_input0[test_columns_None] = _input0[test_columns_None].fillna(_input0[test_columns_None].mode().iloc[0])
test_columns_None = ['MSZoning', 'Functional', 'Utilities', 'Exterior1st', 'Exterior2nd', 'SaleType', 'KitchenQual']
_input0[test_columns_None] = _input0[test_columns_None].fillna(_input0[test_columns_None].mode().iloc[0])
_input0[categorical_cols] = encoder.transform(_input0[categorical_cols])
_input0 = _input0[numerical_cols + categorical_cols]
_input1.SalePrice.describe()
plt.figure(figsize=(10, 5))
sns.histplot(data=_input1, x=_input1['SalePrice'], kde=True)
print('Skewness: %f' % _input1['SalePrice'].skew())
print('Kurtosis: %f' % _input1['SalePrice'].kurt())
_input1['SalePrice'] = np.log1p(_input1['SalePrice'])
_input1.SalePrice.describe()
plt.figure(figsize=(10, 5))
sns.histplot(data=_input1, x=_input1['SalePrice'], kde=True)
print('Skewness: %f' % _input1['SalePrice'].skew())
print('Kurtosis: %f' % _input1['SalePrice'].kurt())
X = _input1[numerical_cols + categorical_cols]
y = _input1.SalePrice
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
fold = 0
for (train_idx, valid_idx) in kf.split(X, y):
    (X_train, X_valid) = (X.iloc[train_idx], X.iloc[valid_idx])
    (y_train, y_valid) = (y.iloc[train_idx], y.iloc[valid_idx])
    dtr = DecisionTreeRegressor()