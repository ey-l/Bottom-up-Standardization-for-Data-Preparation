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
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
sample_submission = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
sns.distplot(train.SalePrice)
sns.distplot(np.log(train.SalePrice + 1))
all_data = pd.concat((train.drop(['SalePrice'], axis=1), test))
all_data_na = all_data.isnull().sum() / len(all_data) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
plt.figure(figsize=(12, 6))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
all_data[all_data.PoolArea != 0][['PoolArea', 'PoolQC']]
all_data[all_data.MiscVal > 10000][['MiscFeature', 'MiscVal']]
all_data[all_data.GarageType.notnull() & all_data.GarageYrBlt.isnull()][['Neighborhood', 'YearBuilt', 'YearRemodAdd', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond']]
train.loc[[332, 948]][['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'BsmtFullBath', 'BsmtHalfBath']]
test.loc[[27, 580, 725, 757, 758, 888, 1064]][['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'BsmtFullBath', 'BsmtHalfBath']]
plt.scatter(train.Utilities, train.SalePrice)
y = train['SalePrice']
y = np.log(y + 1)
test.loc[960, 'PoolQC'] = 'Fa'
test.loc[1043, 'PoolQC'] = 'Gd'
test.loc[1139, 'PoolQC'] = 'Fa'
test.loc[666, 'GarageYrBlt'] = 1979
test.loc[1116, 'GarageYrBlt'] = 1979
test.loc[666, 'GarageFinish'] = 'Unf'
test.loc[1116, 'GarageFinish'] = 'Unf'
test.loc[1116, 'GarageCars'] = 2
test.loc[1116, 'GarageArea'] = 480
test.loc[666, 'GarageQual'] = 'TA'
test.loc[1116, 'GarageQual'] = 'TA'
test.loc[666, 'GarageCond'] = 'TA'
test.loc[1116, 'GarageCond'] = 'TA'
train = train.fillna({'PoolQC': 'None'})
test = test.fillna({'PoolQC': 'None'})
train = train.fillna({'Alley': 'None'})
test = test.fillna({'Alley': 'None'})
train = train.fillna({'FireplaceQu': 'None'})
test = test.fillna({'FireplaceQu': 'None'})
train = train.fillna({'LotFrontage': 0})
test = test.fillna({'LotFrontage': 0})
train = train.fillna({'GarageType': 'None'})
test = test.fillna({'GarageType': 'None'})
train = train.fillna({'GarageYrBlt': 0})
test = test.fillna({'GarageYrBlt': 0})
train = train.fillna({'GarageFinish': 'None'})
test = test.fillna({'GarageFinish': 'None'})
test = test.fillna({'GarageCars': 0})
test = test.fillna({'GarageArea': 0})
train = train.fillna({'GarageQual': 'None'})
test = test.fillna({'GarageQual': 'None'})
train = train.fillna({'GarageCond': 'None'})
test = test.fillna({'GarageCond': 'None'})
train = train.fillna({'BsmtQual': 'None'})
test = test.fillna({'BsmtQual': 'None'})
train = train.fillna({'BsmtCond': 'None'})
test = test.fillna({'BsmtCond': 'None'})
train = train.fillna({'BsmtExposure': 'None'})
test = test.fillna({'BsmtExposure': 'None'})
train = train.fillna({'BsmtFinType1': 'None'})
test = test.fillna({'BsmtFinType1': 'None'})
train = train.fillna({'BsmtFinSF1': 0})
test = test.fillna({'BsmtFinSF1': 0})
train = train.fillna({'BsmtFinType2': 'None'})
test = test.fillna({'BsmtFinType2': 'None'})
test = test.fillna({'BsmtFinSF2': 0})
test = test.fillna({'BsmtUnfSF': 0})
test = test.fillna({'TotalBsmtSF': 0})
test = test.fillna({'BsmtFullBath': 0})
test = test.fillna({'BsmtHalfBath': 0})
train = train.fillna({'MasVnrType': 'None'})
test = test.fillna({'MasVnrType': 'None'})
train = train.fillna({'MasVnrArea': 0})
test = test.fillna({'MasVnrArea': 0})
train = train.drop(['Fence', 'MiscFeature', 'Utilities'], axis=1)
test = test.drop(['Fence', 'MiscFeature', 'Utilities'], axis=1)
test = test.fillna({'MSZoning': 'RL'})
test = test.fillna({'Exterior1st': 'VinylSd'})
test = test.fillna({'Exterior2nd': 'VinylSd'})
train = train.fillna({'Electrical': 'SBrkr'})
test = test.fillna({'KitchenQual': 'TA'})
test = test.fillna({'Functional': 'Typ'})
test = test.fillna({'SaleType': 'WD'})
train_dummies = pd.get_dummies(pd.concat((train.drop(['SalePrice', 'Id'], axis=1), test.drop(['Id'], axis=1)), axis=0)).iloc[:train.shape[0]]
test_dummies = pd.get_dummies(pd.concat((train.drop(['SalePrice', 'Id'], axis=1), test.drop(['Id'], axis=1)), axis=0)).iloc[train.shape[0]:]
rr = Ridge(alpha=10)