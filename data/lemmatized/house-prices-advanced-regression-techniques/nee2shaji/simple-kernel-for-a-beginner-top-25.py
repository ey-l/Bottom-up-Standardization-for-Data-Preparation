import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xgboost as xgb
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.metrics import mean_absolute_error
from scipy.stats import skew
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
all_data = pd.concat((_input1, _input0), sort=False).reset_index(drop=True)
all_data = all_data.drop(['SalePrice'], axis=1, inplace=False)
all_data = all_data.drop(['Id'], axis=1)
cols_with_missing = [col for col in all_data.columns if all_data[col].isnull().any()]
cols_with_missing
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
all_data['Alley'] = all_data['Alley'].fillna('None')
all_data['Utilities'] = all_data['Utilities'].fillna(all_data['Utilities'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['MasVnrType'] = all_data['MasVnrType'].fillna('None')
all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Functional'] = all_data['Functional'].fillna(all_data['Functional'].mode()[0])
all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna('None')
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageCars', 'GarageArea'):
    all_data[col] = all_data[col].fillna(0)
all_data['PoolQC'] = all_data['PoolQC'].fillna('None')
all_data['Fence'] = all_data['Fence'].fillna('None')
all_data['MiscFeature'] = all_data['MiscFeature'].fillna('None')
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
cols_with_missing = [col for col in all_data.columns if all_data[col].isnull().any()]
len(cols_with_missing)
sns.distplot(_input1['SalePrice'])
_input1['SalePrice'] = np.log1p(_input1['SalePrice'])
sns.distplot(_input1['SalePrice'])
numerical_features = all_data.select_dtypes(exclude=['object']).columns
print('Number of numerical features:' + str(len(numerical_features)))
skewness = all_data[numerical_features].apply(lambda x: skew(x))
skewness = skewness[abs(skewness) > 0.7]
skewed_features = skewness.index
all_data[skewed_features] = np.log1p(all_data[skewed_features])
categorical_features = all_data.select_dtypes(include=['object']).columns
print('Number of categorical features:' + str(len(categorical_features)))
dummy_all_data = pd.get_dummies(all_data[categorical_features])
all_data = all_data.drop(categorical_features, axis=1, inplace=False)
all_data = pd.concat([all_data, dummy_all_data], axis=1)
X = all_data[:_input1.shape[0]]
test_data = all_data[_input1.shape[0]:]
y = _input1['SalePrice']

def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5))
    return rmse