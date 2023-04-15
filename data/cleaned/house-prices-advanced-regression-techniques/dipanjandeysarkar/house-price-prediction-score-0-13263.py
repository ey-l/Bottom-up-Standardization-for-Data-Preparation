import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
import xgboost
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
train.head()
train.shape
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
test.shape
train.describe()
train.dtypes
train.dtypes.value_counts()
train['SalePrice'].describe()
unique = len(set(train['Id']))
total = len(train['Id'])
dup = total - unique
print('No of duplicate ID values in train dataset :', dup)
unique_t = len(set(test['Id']))
total_t = len(test['Id'])
dup_t = total_t - unique_t
print('No of duplicate ID values in test dataset :', dup_t)
submission = pd.DataFrame()
submission['Id'] = test['Id']
train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)
misval = train.isnull().sum()
misval = misval[misval > 0]
print(misval)
train.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)
num_feat = train.select_dtypes(exclude=[object]).columns
cat_feat = train.select_dtypes(include=[object]).columns
print('No. of numerical features:', len(num_feat))
print(num_feat)
print('No. of categorical features:', len(cat_feat))
print(cat_feat)
train = train.fillna(train.median())
sns.heatmap(train.isnull(), cbar=False)
misval = train.isnull().sum()
misval = misval[misval > 0]
print(misval)
train['BsmtQual'] = train['BsmtQual'].fillna(train['BsmtQual'].mode()[0])
train['BsmtCond'] = train['BsmtCond'].fillna(train['BsmtCond'].mode()[0])
train['BsmtExposure'] = train['BsmtExposure'].fillna(train['BsmtExposure'].mode()[0])
train['BsmtFinType1'] = train['BsmtFinType1'].fillna(train['BsmtFinType1'].mode()[0])
train['BsmtFinType2'] = train['BsmtFinType2'].fillna(train['BsmtFinType2'].mode()[0])
train['GarageType'] = train['GarageType'].fillna(train['GarageType'].mode()[0])
train['GarageFinish'] = train['GarageFinish'].fillna(train['GarageFinish'].mode()[0])
train['GarageQual'] = train['GarageQual'].fillna(train['GarageQual'].mode()[0])
train['GarageCond'] = train['GarageCond'].fillna(train['GarageCond'].mode()[0])
train['MasVnrType'] = train['MasVnrType'].fillna(train['MasVnrType'].mode()[0])
train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode()[0])
sns.heatmap(train.isnull(), cbar=False)
misval_t = test.isnull().sum()
misval_t = misval_t[misval_t > 0]
print(misval_t)
test.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)
num_feat_test = test.select_dtypes(exclude='object').columns
cat_feat_test = test.select_dtypes(include='object').columns
print('No. of numerical features in test dataset:', len(num_feat_test))
print(num_feat_test)
print('No. of categorical features in test dataset:', len(cat_feat_test))
print(cat_feat_test)
test = test.fillna(test.median())
sns.heatmap(test.isnull(), cbar=False)
test['BsmtQual'] = test['BsmtQual'].fillna(test['BsmtQual'].mode()[0])
test['BsmtCond'] = test['BsmtCond'].fillna(test['BsmtCond'].mode()[0])
test['BsmtExposure'] = test['BsmtExposure'].fillna(test['BsmtExposure'].mode()[0])
test['BsmtFinType1'] = test['BsmtFinType1'].fillna(test['BsmtFinType1'].mode()[0])
test['BsmtFinType2'] = test['BsmtFinType2'].fillna(test['BsmtFinType2'].mode()[0])
test['GarageType'] = test['GarageType'].fillna(test['GarageType'].mode()[0])
test['GarageFinish'] = test['GarageFinish'].fillna(test['GarageFinish'].mode()[0])
test['GarageQual'] = test['GarageQual'].fillna(test['GarageQual'].mode()[0])
test['GarageCond'] = test['GarageCond'].fillna(test['GarageCond'].mode()[0])
test['MasVnrType'] = test['MasVnrType'].fillna(test['MasVnrType'].mode()[0])
misval_t = test.isnull().sum()
misval_t = misval_t[misval_t > 0]
print(misval_t)
test['MSZoning'] = test['MSZoning'].fillna(test['MSZoning'].mode()[0])
test['Utilities'] = test['Utilities'].fillna(test['Utilities'].mode()[0])
test['Exterior1st'] = test['Exterior1st'].fillna(test['Exterior1st'].mode()[0])
test['Exterior2nd'] = test['Exterior2nd'].fillna(test['Exterior2nd'].mode()[0])
test['KitchenQual'] = test['KitchenQual'].fillna(test['KitchenQual'].mode()[0])
test['Functional'] = test['Functional'].fillna(test['Functional'].mode()[0])
test['SaleType'] = test['SaleType'].fillna(test['SaleType'].mode()[0])
sns.heatmap(test.isnull(), cbar=False)
test.shape
train.shape
file = pd.concat([train, test], axis=0)
file.shape
file = pd.get_dummies(file)
file.shape
file.head()
mis_file = file.isnull().sum()
mis_file = mis_file[mis_file > 0]
print(mis_file)
traindata = file.iloc[:1460]
traindata.shape
traindata.head()
testdata = file.iloc[1460:]
testdata.drop(['SalePrice'], axis=1, inplace=True)
testdata.shape
sns.distplot(traindata['SalePrice'])
traindata['SalePrice'] = np.log(traindata['SalePrice'])
sns.distplot(traindata['SalePrice'])
x_train = traindata.drop(['SalePrice'], axis=1)
y_train = traindata['SalePrice']
scaler = StandardScaler()
scaler.fit_transform(x_train)
model = Lasso()
param_grid = {'alpha': [0.0001, 0.01, 1, 10, 100]}
grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=5)