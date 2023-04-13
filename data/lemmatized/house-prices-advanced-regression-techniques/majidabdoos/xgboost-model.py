import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
test_ID = _input0['Id']
_input1 = _input1.drop(columns='Id')
final_test_Id = _input0['Id']
_input0 = _input0.drop(columns='Id')
np.abs(_input1.corr()['SalePrice']).sort_values(ascending=False)
(fig, axs) = plt.subplots(ncols=2, figsize=(17, 5))
sns.scatterplot(data=_input1, y='SalePrice', x='OverallQual', ax=axs[0])
sns.scatterplot(data=_input1, y='SalePrice', x='GrLivArea', ax=axs[1])
_input1 = _input1[~((_input1['OverallQual'] == 10) & (_input1['SalePrice'] < 200000))]

def missing_data():
    df_missing = _input1.isnull().sum()
    df_missing = df_missing[df_missing > 0].sort_values(ascending=False)
    test_missing = _input0.isnull().sum()
    test_missing = test_missing[test_missing > 0].sort_values(ascending=False)
    missing = pd.DataFrame(data=[df_missing, test_missing], index=['Train', 'Test']).T
    return missing
missing_data().plot(kind='bar', figsize=(15, 6))
features_cat = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
_input1[features_cat] = _input1[features_cat].fillna('None')
_input0[features_cat] = _input0[features_cat].fillna('None')
_input1['LotFrontage'] = _input1.groupby('Neighborhood')['LotFrontage'].transform(lambda val: val.fillna(val.mean()))
_input0['LotFrontage'] = _input0.groupby('Neighborhood')['LotFrontage'].transform(lambda val: val.fillna(val.mean()))
features_cat = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
_input1[features_cat] = _input1[features_cat].fillna('None')
_input0[features_cat] = _input0[features_cat].fillna('None')
features_num = ['GarageYrBlt', 'GarageArea', 'GarageCars']
_input1[features_num] = _input1[features_num].fillna(0)
_input0[features_num] = _input0[features_num].fillna(0)
features_cat = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
_input1[features_cat] = _input1[features_cat].fillna('None')
_input0[features_cat] = _input0[features_cat].fillna('None')
features_num = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']
_input1[features_num] = _input1[features_num].fillna(0)
_input0[features_num] = _input0[features_num].fillna(0)
_input1['MasVnrType'] = _input1['MasVnrType'].fillna('None')
_input0['MasVnrType'] = _input0['MasVnrType'].fillna('None')
_input1['MasVnrArea'] = _input1['MasVnrArea'].fillna(0)
_input0['MasVnrArea'] = _input0['MasVnrArea'].fillna(0)
_input1 = _input1[~_input1['Electrical'].isnull()]
_input0['MSZoning'] = _input0['MSZoning'].fillna(_input0['MSZoning'].mode()[0])
_input1 = _input1.drop(columns='Utilities')
_input0 = _input0.drop(columns='Utilities')
_input0['Functional'] = _input0['Functional'].fillna(_input0['Functional'].mode()[0])
_input0['Exterior1st'] = _input0['Exterior1st'].fillna(_input0['Exterior1st'].mode()[0])
_input0['Exterior2nd'] = _input0['Exterior2nd'].fillna(_input0['Exterior2nd'].mode()[0])
_input0['KitchenQual'] = _input0['KitchenQual'].fillna(_input0['KitchenQual'].mode()[0])
_input0['SaleType'] = _input0['SaleType'].fillna(_input0['SaleType'].mode()[0])
missing_data()
n_train = _input1.shape[0]
X = _input1.drop(columns='SalePrice')
y_train = _input1['SalePrice']
all_data = pd.concat((X, _input0))
features = ['MSSubClass', 'MoSold', 'YrSold']
all_data[features] = all_data[features].astype('object')
all_data = pd.get_dummies(all_data)
X_train = all_data[:n_train]
X_test = all_data[n_train:]
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
xgb_model = XGBRegressor()
param = {'n_estimators': [50, 100, 200, 400, 800, 1000, 2000], 'learning_rate': [0.2, 0.1, 0.05, 0.01, 0.001], 'max_depth': [1, 2, 3, 5, 6, 7, 8, 9], 'min_child_weight': [0.5, 1, 3, 5, 8, 10], 'gamma': [50, 100, 120, 150, 180, 200], 'reg_lambda': [0, 1, 5, 10]}
rand_search = RandomizedSearchCV(xgb_model, param)