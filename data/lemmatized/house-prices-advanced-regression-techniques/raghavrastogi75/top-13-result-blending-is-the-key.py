import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input2 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.columns
_input1.info()
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
plt.scatter(_input1['GrLivArea'], _input1['SalePrice'])
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Area vs price')
_input1 = _input1.loc[(_input1['GrLivArea'] < 4000) | (_input1['SalePrice'] > 300000)]
_input1['SalePrice'] = np.log1p(_input1['SalePrice'])
y = _input1.SalePrice
_input1.loc[:, 'Alley'] = _input1.loc[:, 'Alley'].fillna('None')
_input1.loc[:, 'BedroomAbvGr'] = _input1.loc[:, 'BedroomAbvGr'].fillna(0)
_input1.loc[:, 'BsmtQual'] = _input1.loc[:, 'BsmtQual'].fillna('No')
_input1.loc[:, 'BsmtCond'] = _input1.loc[:, 'BsmtCond'].fillna('No')
_input1.loc[:, 'BsmtExposure'] = _input1.loc[:, 'BsmtExposure'].fillna('No')
_input1.loc[:, 'BsmtFinType1'] = _input1.loc[:, 'BsmtFinType1'].fillna('No')
_input1.loc[:, 'BsmtFinType2'] = _input1.loc[:, 'BsmtFinType2'].fillna('No')
_input1.loc[:, 'BsmtFullBath'] = _input1.loc[:, 'BsmtFullBath'].fillna(0)
_input1.loc[:, 'BsmtHalfBath'] = _input1.loc[:, 'BsmtHalfBath'].fillna(0)
_input1.loc[:, 'BsmtUnfSF'] = _input1.loc[:, 'BsmtUnfSF'].fillna(0)
_input1.loc[:, 'CentralAir'] = _input1.loc[:, 'CentralAir'].fillna('N')
_input1.loc[:, 'Condition1'] = _input1.loc[:, 'Condition1'].fillna('Norm')
_input1.loc[:, 'Condition2'] = _input1.loc[:, 'Condition2'].fillna('Norm')
_input1.loc[:, 'EnclosedPorch'] = _input1.loc[:, 'EnclosedPorch'].fillna(0)
_input1.loc[:, 'ExterCond'] = _input1.loc[:, 'ExterCond'].fillna('TA')
_input1.loc[:, 'ExterQual'] = _input1.loc[:, 'ExterQual'].fillna('TA')
_input1.loc[:, 'Fence'] = _input1.loc[:, 'Fence'].fillna('No')
_input1.loc[:, 'FireplaceQu'] = _input1.loc[:, 'FireplaceQu'].fillna('No')
_input1.loc[:, 'Fireplaces'] = _input1.loc[:, 'Fireplaces'].fillna(0)
_input1.loc[:, 'Functional'] = _input1.loc[:, 'Functional'].fillna('Typ')
_input1.loc[:, 'GarageType'] = _input1.loc[:, 'GarageType'].fillna('No')
_input1.loc[:, 'GarageFinish'] = _input1.loc[:, 'GarageFinish'].fillna('No')
_input1.loc[:, 'GarageQual'] = _input1.loc[:, 'GarageQual'].fillna('No')
_input1.loc[:, 'GarageCond'] = _input1.loc[:, 'GarageCond'].fillna('No')
_input1.loc[:, 'GarageArea'] = _input1.loc[:, 'GarageArea'].fillna(0)
_input1.loc[:, 'GarageCars'] = _input1.loc[:, 'GarageCars'].fillna(0)
_input1.loc[:, 'HalfBath'] = _input1.loc[:, 'HalfBath'].fillna(0)
_input1.loc[:, 'HeatingQC'] = _input1.loc[:, 'HeatingQC'].fillna('TA')
_input1.loc[:, 'KitchenAbvGr'] = _input1.loc[:, 'KitchenAbvGr'].fillna(0)
_input1.loc[:, 'KitchenQual'] = _input1.loc[:, 'KitchenQual'].fillna('TA')
_input1.loc[:, 'LotFrontage'] = _input1.loc[:, 'LotFrontage'].fillna(0)
_input1.loc[:, 'LotShape'] = _input1.loc[:, 'LotShape'].fillna('Reg')
_input1.loc[:, 'MasVnrType'] = _input1.loc[:, 'MasVnrType'].fillna('None')
_input1.loc[:, 'MasVnrArea'] = _input1.loc[:, 'MasVnrArea'].fillna(0)
_input1.loc[:, 'MiscFeature'] = _input1.loc[:, 'MiscFeature'].fillna('No')
_input1.loc[:, 'MiscVal'] = _input1.loc[:, 'MiscVal'].fillna(0)
_input1.loc[:, 'OpenPorchSF'] = _input1.loc[:, 'OpenPorchSF'].fillna(0)
_input1.loc[:, 'PavedDrive'] = _input1.loc[:, 'PavedDrive'].fillna('N')
_input1.loc[:, 'PoolQC'] = _input1.loc[:, 'PoolQC'].fillna('No')
_input1.loc[:, 'PoolArea'] = _input1.loc[:, 'PoolArea'].fillna(0)
_input1.loc[:, 'SaleCondition'] = _input1.loc[:, 'SaleCondition'].fillna('Normal')
_input1.loc[:, 'ScreenPorch'] = _input1.loc[:, 'ScreenPorch'].fillna(0)
_input1.loc[:, 'TotRmsAbvGrd'] = _input1.loc[:, 'TotRmsAbvGrd'].fillna(0)
_input1.loc[:, 'Utilities'] = _input1.loc[:, 'Utilities'].fillna('AllPub')
_input1.loc[:, 'WoodDeckSF'] = _input1.loc[:, 'WoodDeckSF'].fillna(0)
_input1 = _input1.replace({'MSSubClass': {20: 'SC20', 30: 'SC30', 40: 'SC40', 45: 'SC45', 50: 'SC50', 60: 'SC60', 70: 'SC70', 75: 'SC75', 80: 'SC80', 85: 'SC85', 90: 'SC90', 120: 'SC120', 150: 'SC150', 160: 'SC160', 180: 'SC180', 190: 'SC190'}, 'MoSold': {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}})
_input1 = _input1.replace({'Alley': {'Grvl': 1, 'Pave': 2}, 'BsmtCond': {'No': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'BsmtExposure': {'No': 0, 'Mn': 1, 'Av': 2, 'Gd': 3}, 'BsmtFinType1': {'No': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}, 'BsmtFinType2': {'No': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}, 'BsmtQual': {'No': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'ExterCond': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'ExterQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'FireplaceQu': {'No': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'Functional': {'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8}, 'GarageCond': {'No': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'GarageQual': {'No': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'HeatingQC': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'KitchenQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'LandSlope': {'Sev': 1, 'Mod': 2, 'Gtl': 3}, 'LotShape': {'IR3': 1, 'IR2': 2, 'IR1': 3, 'Reg': 4}, 'PavedDrive': {'N': 0, 'P': 1, 'Y': 2}, 'PoolQC': {'No': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}, 'Street': {'Grvl': 1, 'Pave': 2}, 'Utilities': {'ELO': 1, 'NoSeWa': 2, 'NoSewr': 3, 'AllPub': 4}})
_input1['SimplOverallQual'] = _input1.OverallQual.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 3})
_input1['SimplOverallCond'] = _input1.OverallCond.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 3})
_input1['SimplPoolQC'] = _input1.PoolQC.replace({1: 1, 2: 1, 3: 2, 4: 2})
_input1['SimplGarageCond'] = _input1.GarageCond.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
_input1['SimplGarageQual'] = _input1.GarageQual.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
_input1['SimplFireplaceQu'] = _input1.FireplaceQu.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
_input1['SimplFireplaceQu'] = _input1.FireplaceQu.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
_input1['SimplFunctional'] = _input1.Functional.replace({1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 3, 8: 4})
_input1['SimplKitchenQual'] = _input1.KitchenQual.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
_input1['SimplHeatingQC'] = _input1.HeatingQC.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
_input1['SimplBsmtFinType1'] = _input1.BsmtFinType1.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2})
_input1['SimplBsmtFinType2'] = _input1.BsmtFinType2.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2})
_input1['SimplBsmtCond'] = _input1.BsmtCond.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
_input1['SimplBsmtQual'] = _input1.BsmtQual.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
_input1['SimplExterCond'] = _input1.ExterCond.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
_input1['SimplExterQual'] = _input1.ExterQual.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
_input1['OverallGrade'] = _input1['OverallQual'] * _input1['OverallCond']
_input1['GarageGrade'] = _input1['GarageQual'] * _input1['GarageCond']
_input1['ExterGrade'] = _input1['ExterQual'] * _input1['ExterCond']
_input1['KitchenScore'] = _input1['KitchenAbvGr'] * _input1['KitchenQual']
_input1['FireplaceScore'] = _input1['Fireplaces'] * _input1['FireplaceQu']
_input1['GarageScore'] = _input1['GarageArea'] * _input1['GarageQual']
_input1['PoolScore'] = _input1['PoolArea'] * _input1['PoolQC']
_input1['SimplOverallGrade'] = _input1['SimplOverallQual'] * _input1['SimplOverallCond']
_input1['SimplExterGrade'] = _input1['SimplExterQual'] * _input1['SimplExterCond']
_input1['SimplPoolScore'] = _input1['PoolArea'] * _input1['SimplPoolQC']
_input1['SimplGarageScore'] = _input1['GarageArea'] * _input1['SimplGarageQual']
_input1['SimplFireplaceScore'] = _input1['Fireplaces'] * _input1['SimplFireplaceQu']
_input1['SimplKitchenScore'] = _input1['KitchenAbvGr'] * _input1['SimplKitchenQual']
_input1['TotalBath'] = _input1['BsmtFullBath'] + 0.5 * _input1['BsmtHalfBath'] + _input1['FullBath'] + 0.5 * _input1['HalfBath']
_input1['AllSF'] = _input1['GrLivArea'] + _input1['TotalBsmtSF']
_input1['AllFlrsSF'] = _input1['1stFlrSF'] + _input1['2ndFlrSF']
_input1['AllPorchSF'] = _input1['OpenPorchSF'] + _input1['EnclosedPorch'] + _input1['3SsnPorch'] + _input1['ScreenPorch']
_input1['HasMasVnr'] = _input1.MasVnrType.replace({'BrkCmn': 1, 'BrkFace': 1, 'CBlock': 1, 'Stone': 1, 'None': 0})
_input1['BoughtOffPlan'] = _input1.SaleCondition.replace({'Abnorml': 0, 'Alloca': 0, 'AdjLand': 0, 'Family': 0, 'Normal': 0, 'Partial': 1})
print('Find most important features relative to target')
corr = _input1.corr()
corr = corr.sort_values(['SalePrice'], ascending=False, inplace=False)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(corr.SalePrice)
_input1['OverallQual-s2'] = _input1['OverallQual'] ** 2
_input1['OverallQual-s3'] = _input1['OverallQual'] ** 3
_input1['OverallQual-Sq'] = np.sqrt(_input1['OverallQual'])
_input1['AllSF-2'] = _input1['AllSF'] ** 2
_input1['AllSF-3'] = _input1['AllSF'] ** 3
_input1['AllSF-Sq'] = np.sqrt(_input1['AllSF'])
_input1['AllFlrsSF-2'] = _input1['AllFlrsSF'] ** 2
_input1['AllFlrsSF-3'] = _input1['AllFlrsSF'] ** 3
_input1['AllFlrsSF-Sq'] = np.sqrt(_input1['AllFlrsSF'])
_input1['GrLivArea-2'] = _input1['GrLivArea'] ** 2
_input1['GrLivArea-3'] = _input1['GrLivArea'] ** 3
_input1['GrLivArea-Sq'] = np.sqrt(_input1['GrLivArea'])
_input1['SimplOverallQual-s2'] = _input1['SimplOverallQual'] ** 2
_input1['SimplOverallQual-s3'] = _input1['SimplOverallQual'] ** 3
_input1['SimplOverallQual-Sq'] = np.sqrt(_input1['SimplOverallQual'])
_input1['ExterQual-2'] = _input1['ExterQual'] ** 2
_input1['ExterQual-3'] = _input1['ExterQual'] ** 3
_input1['ExterQual-Sq'] = np.sqrt(_input1['ExterQual'])
_input1['GarageCars-2'] = _input1['GarageCars'] ** 2
_input1['GarageCars-3'] = _input1['GarageCars'] ** 3
_input1['GarageCars-Sq'] = np.sqrt(_input1['GarageCars'])
_input1['TotalBath-2'] = _input1['TotalBath'] ** 2
_input1['TotalBath-3'] = _input1['TotalBath'] ** 3
_input1['TotalBath-Sq'] = np.sqrt(_input1['TotalBath'])
_input1['KitchenQual-2'] = _input1['KitchenQual'] ** 2
_input1['KitchenQual-3'] = _input1['KitchenQual'] ** 3
_input1['KitchenQual-Sq'] = np.sqrt(_input1['KitchenQual'])
_input1['GarageScore-2'] = _input1['GarageScore'] ** 2
_input1['GarageScore-3'] = _input1['GarageScore'] ** 3
_input1['GarageScore-Sq'] = np.sqrt(_input1['GarageScore'])
cat_features = _input1.select_dtypes(include=['object']).columns
num_features = _input1.select_dtypes(exclude=['object']).columns
num_features = num_features.drop('SalePrice')
train_num = _input1[num_features]
train_cat = _input1[cat_features]
print('before', train_num.isnull().values.sum())
train_num = train_num.fillna(train_num.median())
print('after', train_num.isnull().values.sum())
from scipy.stats import skew
skewness = train_num.apply(lambda x: skew(x))
skewness = skewness[abs(skewness) > 0.5]
skewed_features = skewness.index
train_num[skewed_features] = np.log1p(train_num[skewed_features])
train_cat = pd.get_dummies(train_cat)
print(_input1)
from sklearn.model_selection import train_test_split
_input1 = pd.concat([train_num, train_cat], axis=1)
(X_train, X_test, y_train, y_test) = train_test_split(_input1, y, test_size=0.3, random_state=42)
from sklearn.preprocessing import StandardScaler
stdSc = StandardScaler()
X_train.loc[:, num_features] = stdSc.fit_transform(X_train.loc[:, num_features])
X_test.loc[:, num_features] = stdSc.transform(X_test.loc[:, num_features])
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from mlxtend.regressor import StackingCVRegressor
from sklearn.model_selection import KFold, cross_val_score
ridge = Ridge()
lasso = Lasso()
elasticnet = ElasticNet()
gbr = GradientBoostingRegressor()
lightgbm = LGBMRegressor()
xgboost = XGBRegressor()
scvr = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, gbr, xgboost, lightgbm), meta_regressor=xgboost, use_features_in_secondary=True)
lasso = Lasso(alpha=0.0005)
ridge = Ridge(alpha=11.9)
elasticnet = ElasticNet(alpha=0.001)
gbr = GradientBoostingRegressor(learning_rate=0.1, loss='huber', max_depth=1, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, n_estimators=3000)
lightgbm = LGBMRegressor(learning_rate=0.01, n_estimators=5000, num_leaves=4)
xgboost = XGBRegressor(learning_rate=0.01, max_depth=3, n_estimators=3500)
scvr = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, gbr, xgboost, lightgbm), meta_regressor=xgboost, use_features_in_secondary=True)
_input1.loc[:, num_features] = stdSc.fit_transform(_input1.loc[:, num_features])
rmse_list = []
models = [lasso, ridge, elasticnet, gbr, lightgbm, xgboost]