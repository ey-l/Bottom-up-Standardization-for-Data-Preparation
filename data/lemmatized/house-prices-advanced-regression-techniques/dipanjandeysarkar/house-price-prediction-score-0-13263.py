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
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1.head()
_input1.shape
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input0.shape
_input1.describe()
_input1.dtypes
_input1.dtypes.value_counts()
_input1['SalePrice'].describe()
unique = len(set(_input1['Id']))
total = len(_input1['Id'])
dup = total - unique
print('No of duplicate ID values in train dataset :', dup)
unique_t = len(set(_input0['Id']))
total_t = len(_input0['Id'])
dup_t = total_t - unique_t
print('No of duplicate ID values in test dataset :', dup_t)
submission = pd.DataFrame()
submission['Id'] = _input0['Id']
_input1 = _input1.drop(['Id'], axis=1, inplace=False)
_input0 = _input0.drop(['Id'], axis=1, inplace=False)
misval = _input1.isnull().sum()
misval = misval[misval > 0]
print(misval)
_input1 = _input1.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=False)
num_feat = _input1.select_dtypes(exclude=[object]).columns
cat_feat = _input1.select_dtypes(include=[object]).columns
print('No. of numerical features:', len(num_feat))
print(num_feat)
print('No. of categorical features:', len(cat_feat))
print(cat_feat)
_input1 = _input1.fillna(_input1.median())
sns.heatmap(_input1.isnull(), cbar=False)
misval = _input1.isnull().sum()
misval = misval[misval > 0]
print(misval)
_input1['BsmtQual'] = _input1['BsmtQual'].fillna(_input1['BsmtQual'].mode()[0])
_input1['BsmtCond'] = _input1['BsmtCond'].fillna(_input1['BsmtCond'].mode()[0])
_input1['BsmtExposure'] = _input1['BsmtExposure'].fillna(_input1['BsmtExposure'].mode()[0])
_input1['BsmtFinType1'] = _input1['BsmtFinType1'].fillna(_input1['BsmtFinType1'].mode()[0])
_input1['BsmtFinType2'] = _input1['BsmtFinType2'].fillna(_input1['BsmtFinType2'].mode()[0])
_input1['GarageType'] = _input1['GarageType'].fillna(_input1['GarageType'].mode()[0])
_input1['GarageFinish'] = _input1['GarageFinish'].fillna(_input1['GarageFinish'].mode()[0])
_input1['GarageQual'] = _input1['GarageQual'].fillna(_input1['GarageQual'].mode()[0])
_input1['GarageCond'] = _input1['GarageCond'].fillna(_input1['GarageCond'].mode()[0])
_input1['MasVnrType'] = _input1['MasVnrType'].fillna(_input1['MasVnrType'].mode()[0])
_input1['Electrical'] = _input1['Electrical'].fillna(_input1['Electrical'].mode()[0])
sns.heatmap(_input1.isnull(), cbar=False)
misval_t = _input0.isnull().sum()
misval_t = misval_t[misval_t > 0]
print(misval_t)
_input0 = _input0.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=False)
num_feat_test = _input0.select_dtypes(exclude='object').columns
cat_feat_test = _input0.select_dtypes(include='object').columns
print('No. of numerical features in test dataset:', len(num_feat_test))
print(num_feat_test)
print('No. of categorical features in test dataset:', len(cat_feat_test))
print(cat_feat_test)
_input0 = _input0.fillna(_input0.median())
sns.heatmap(_input0.isnull(), cbar=False)
_input0['BsmtQual'] = _input0['BsmtQual'].fillna(_input0['BsmtQual'].mode()[0])
_input0['BsmtCond'] = _input0['BsmtCond'].fillna(_input0['BsmtCond'].mode()[0])
_input0['BsmtExposure'] = _input0['BsmtExposure'].fillna(_input0['BsmtExposure'].mode()[0])
_input0['BsmtFinType1'] = _input0['BsmtFinType1'].fillna(_input0['BsmtFinType1'].mode()[0])
_input0['BsmtFinType2'] = _input0['BsmtFinType2'].fillna(_input0['BsmtFinType2'].mode()[0])
_input0['GarageType'] = _input0['GarageType'].fillna(_input0['GarageType'].mode()[0])
_input0['GarageFinish'] = _input0['GarageFinish'].fillna(_input0['GarageFinish'].mode()[0])
_input0['GarageQual'] = _input0['GarageQual'].fillna(_input0['GarageQual'].mode()[0])
_input0['GarageCond'] = _input0['GarageCond'].fillna(_input0['GarageCond'].mode()[0])
_input0['MasVnrType'] = _input0['MasVnrType'].fillna(_input0['MasVnrType'].mode()[0])
misval_t = _input0.isnull().sum()
misval_t = misval_t[misval_t > 0]
print(misval_t)
_input0['MSZoning'] = _input0['MSZoning'].fillna(_input0['MSZoning'].mode()[0])
_input0['Utilities'] = _input0['Utilities'].fillna(_input0['Utilities'].mode()[0])
_input0['Exterior1st'] = _input0['Exterior1st'].fillna(_input0['Exterior1st'].mode()[0])
_input0['Exterior2nd'] = _input0['Exterior2nd'].fillna(_input0['Exterior2nd'].mode()[0])
_input0['KitchenQual'] = _input0['KitchenQual'].fillna(_input0['KitchenQual'].mode()[0])
_input0['Functional'] = _input0['Functional'].fillna(_input0['Functional'].mode()[0])
_input0['SaleType'] = _input0['SaleType'].fillna(_input0['SaleType'].mode()[0])
sns.heatmap(_input0.isnull(), cbar=False)
_input0.shape
_input1.shape
file = pd.concat([_input1, _input0], axis=0)
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
testdata = testdata.drop(['SalePrice'], axis=1, inplace=False)
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