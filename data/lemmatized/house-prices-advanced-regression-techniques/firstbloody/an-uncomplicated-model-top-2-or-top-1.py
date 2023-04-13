import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost.sklearn import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input2 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
sns.distplot(_input1.SalePrice)
sns.distplot(np.log(_input1.SalePrice + 1))
all_data = pd.concat((_input1.drop(['SalePrice'], axis=1), _input0))
all_data_na = all_data.isnull().sum() / len(all_data) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
plt.figure(figsize=(12, 6))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
all_data[all_data.PoolArea != 0][['PoolArea', 'PoolQC']]
all_data[all_data.MiscVal > 10000][['MiscFeature', 'MiscVal']]
all_data[all_data.GarageType.notnull() & all_data.GarageYrBlt.isnull()][['Neighborhood', 'YearBuilt', 'YearRemodAdd', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond']]
_input1.loc[[332, 948]][['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'BsmtFullBath', 'BsmtHalfBath']]
_input0.loc[[27, 580, 725, 757, 758, 888, 1064]][['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'BsmtFullBath', 'BsmtHalfBath']]
plt.scatter(_input1.Utilities, _input1.SalePrice)
y = _input1['SalePrice']
y = np.log(y + 1)
_input0.loc[960, 'PoolQC'] = 'Fa'
_input0.loc[1043, 'PoolQC'] = 'Gd'
_input0.loc[1139, 'PoolQC'] = 'Fa'
_input0.loc[666, 'GarageYrBlt'] = 1979
_input0.loc[1116, 'GarageYrBlt'] = 1979
_input0.loc[666, 'GarageFinish'] = 'Unf'
_input0.loc[1116, 'GarageFinish'] = 'Unf'
_input0.loc[1116, 'GarageCars'] = 2
_input0.loc[1116, 'GarageArea'] = 480
_input0.loc[666, 'GarageQual'] = 'TA'
_input0.loc[1116, 'GarageQual'] = 'TA'
_input0.loc[666, 'GarageCond'] = 'TA'
_input0.loc[1116, 'GarageCond'] = 'TA'
_input1 = _input1.fillna({'PoolQC': 'None'})
_input0 = _input0.fillna({'PoolQC': 'None'})
_input1 = _input1.fillna({'Alley': 'None'})
_input0 = _input0.fillna({'Alley': 'None'})
_input1 = _input1.fillna({'FireplaceQu': 'None'})
_input0 = _input0.fillna({'FireplaceQu': 'None'})
_input1 = _input1.fillna({'LotFrontage': 0})
_input0 = _input0.fillna({'LotFrontage': 0})
_input1 = _input1.fillna({'GarageType': 'None'})
_input0 = _input0.fillna({'GarageType': 'None'})
_input1 = _input1.fillna({'GarageYrBlt': 0})
_input0 = _input0.fillna({'GarageYrBlt': 0})
_input1 = _input1.fillna({'GarageFinish': 'None'})
_input0 = _input0.fillna({'GarageFinish': 'None'})
_input0 = _input0.fillna({'GarageCars': 0})
_input0 = _input0.fillna({'GarageArea': 0})
_input1 = _input1.fillna({'GarageQual': 'None'})
_input0 = _input0.fillna({'GarageQual': 'None'})
_input1 = _input1.fillna({'GarageCond': 'None'})
_input0 = _input0.fillna({'GarageCond': 'None'})
_input1 = _input1.fillna({'BsmtQual': 'None'})
_input0 = _input0.fillna({'BsmtQual': 'None'})
_input1 = _input1.fillna({'BsmtCond': 'None'})
_input0 = _input0.fillna({'BsmtCond': 'None'})
_input1 = _input1.fillna({'BsmtExposure': 'None'})
_input0 = _input0.fillna({'BsmtExposure': 'None'})
_input1 = _input1.fillna({'BsmtFinType1': 'None'})
_input0 = _input0.fillna({'BsmtFinType1': 'None'})
_input1 = _input1.fillna({'BsmtFinSF1': 0})
_input0 = _input0.fillna({'BsmtFinSF1': 0})
_input1 = _input1.fillna({'BsmtFinType2': 'None'})
_input0 = _input0.fillna({'BsmtFinType2': 'None'})
_input0 = _input0.fillna({'BsmtFinSF2': 0})
_input0 = _input0.fillna({'BsmtUnfSF': 0})
_input0 = _input0.fillna({'TotalBsmtSF': 0})
_input0 = _input0.fillna({'BsmtFullBath': 0})
_input0 = _input0.fillna({'BsmtHalfBath': 0})
_input1 = _input1.fillna({'MasVnrType': 'None'})
_input0 = _input0.fillna({'MasVnrType': 'None'})
_input1 = _input1.fillna({'MasVnrArea': 0})
_input0 = _input0.fillna({'MasVnrArea': 0})
_input1 = _input1.drop(['Fence', 'MiscFeature', 'Utilities'], axis=1)
_input0 = _input0.drop(['Fence', 'MiscFeature', 'Utilities'], axis=1)
_input0 = _input0.fillna({'MSZoning': 'RL'})
_input0 = _input0.fillna({'Exterior1st': 'VinylSd'})
_input0 = _input0.fillna({'Exterior2nd': 'VinylSd'})
_input1 = _input1.fillna({'Electrical': 'SBrkr'})
_input0 = _input0.fillna({'KitchenQual': 'TA'})
_input0 = _input0.fillna({'Functional': 'Typ'})
_input0 = _input0.fillna({'SaleType': 'WD'})
train_dummies = pd.get_dummies(pd.concat((_input1.drop(['SalePrice', 'Id'], axis=1), _input0.drop(['Id'], axis=1)), axis=0)).iloc[:_input1.shape[0]]
test_dummies = pd.get_dummies(pd.concat((_input1.drop(['SalePrice', 'Id'], axis=1), _input0.drop(['Id'], axis=1)), axis=0)).iloc[_input1.shape[0]:]
rr = Ridge(alpha=10)