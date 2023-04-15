import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
sample = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train.columns
train.info()
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
plt.scatter(train['GrLivArea'], train['SalePrice'])
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Area vs price')
train = train.loc[(train['GrLivArea'] < 4000) | (train['SalePrice'] > 300000)]
train['SalePrice'] = np.log1p(train['SalePrice'])
y = train.SalePrice
train.loc[:, 'Alley'] = train.loc[:, 'Alley'].fillna('None')
train.loc[:, 'BedroomAbvGr'] = train.loc[:, 'BedroomAbvGr'].fillna(0)
train.loc[:, 'BsmtQual'] = train.loc[:, 'BsmtQual'].fillna('No')
train.loc[:, 'BsmtCond'] = train.loc[:, 'BsmtCond'].fillna('No')
train.loc[:, 'BsmtExposure'] = train.loc[:, 'BsmtExposure'].fillna('No')
train.loc[:, 'BsmtFinType1'] = train.loc[:, 'BsmtFinType1'].fillna('No')
train.loc[:, 'BsmtFinType2'] = train.loc[:, 'BsmtFinType2'].fillna('No')
train.loc[:, 'BsmtFullBath'] = train.loc[:, 'BsmtFullBath'].fillna(0)
train.loc[:, 'BsmtHalfBath'] = train.loc[:, 'BsmtHalfBath'].fillna(0)
train.loc[:, 'BsmtUnfSF'] = train.loc[:, 'BsmtUnfSF'].fillna(0)
train.loc[:, 'CentralAir'] = train.loc[:, 'CentralAir'].fillna('N')
train.loc[:, 'Condition1'] = train.loc[:, 'Condition1'].fillna('Norm')
train.loc[:, 'Condition2'] = train.loc[:, 'Condition2'].fillna('Norm')
train.loc[:, 'EnclosedPorch'] = train.loc[:, 'EnclosedPorch'].fillna(0)
train.loc[:, 'ExterCond'] = train.loc[:, 'ExterCond'].fillna('TA')
train.loc[:, 'ExterQual'] = train.loc[:, 'ExterQual'].fillna('TA')
train.loc[:, 'Fence'] = train.loc[:, 'Fence'].fillna('No')
train.loc[:, 'FireplaceQu'] = train.loc[:, 'FireplaceQu'].fillna('No')
train.loc[:, 'Fireplaces'] = train.loc[:, 'Fireplaces'].fillna(0)
train.loc[:, 'Functional'] = train.loc[:, 'Functional'].fillna('Typ')
train.loc[:, 'GarageType'] = train.loc[:, 'GarageType'].fillna('No')
train.loc[:, 'GarageFinish'] = train.loc[:, 'GarageFinish'].fillna('No')
train.loc[:, 'GarageQual'] = train.loc[:, 'GarageQual'].fillna('No')
train.loc[:, 'GarageCond'] = train.loc[:, 'GarageCond'].fillna('No')
train.loc[:, 'GarageArea'] = train.loc[:, 'GarageArea'].fillna(0)
train.loc[:, 'GarageCars'] = train.loc[:, 'GarageCars'].fillna(0)
train.loc[:, 'HalfBath'] = train.loc[:, 'HalfBath'].fillna(0)
train.loc[:, 'HeatingQC'] = train.loc[:, 'HeatingQC'].fillna('TA')
train.loc[:, 'KitchenAbvGr'] = train.loc[:, 'KitchenAbvGr'].fillna(0)
train.loc[:, 'KitchenQual'] = train.loc[:, 'KitchenQual'].fillna('TA')
train.loc[:, 'LotFrontage'] = train.loc[:, 'LotFrontage'].fillna(0)
train.loc[:, 'LotShape'] = train.loc[:, 'LotShape'].fillna('Reg')
train.loc[:, 'MasVnrType'] = train.loc[:, 'MasVnrType'].fillna('None')
train.loc[:, 'MasVnrArea'] = train.loc[:, 'MasVnrArea'].fillna(0)
train.loc[:, 'MiscFeature'] = train.loc[:, 'MiscFeature'].fillna('No')
train.loc[:, 'MiscVal'] = train.loc[:, 'MiscVal'].fillna(0)
train.loc[:, 'OpenPorchSF'] = train.loc[:, 'OpenPorchSF'].fillna(0)
train.loc[:, 'PavedDrive'] = train.loc[:, 'PavedDrive'].fillna('N')
train.loc[:, 'PoolQC'] = train.loc[:, 'PoolQC'].fillna('No')
train.loc[:, 'PoolArea'] = train.loc[:, 'PoolArea'].fillna(0)
train.loc[:, 'SaleCondition'] = train.loc[:, 'SaleCondition'].fillna('Normal')
train.loc[:, 'ScreenPorch'] = train.loc[:, 'ScreenPorch'].fillna(0)
train.loc[:, 'TotRmsAbvGrd'] = train.loc[:, 'TotRmsAbvGrd'].fillna(0)
train.loc[:, 'Utilities'] = train.loc[:, 'Utilities'].fillna('AllPub')
train.loc[:, 'WoodDeckSF'] = train.loc[:, 'WoodDeckSF'].fillna(0)
train = train.replace({'MSSubClass': {20: 'SC20', 30: 'SC30', 40: 'SC40', 45: 'SC45', 50: 'SC50', 60: 'SC60', 70: 'SC70', 75: 'SC75', 80: 'SC80', 85: 'SC85', 90: 'SC90', 120: 'SC120', 150: 'SC150', 160: 'SC160', 180: 'SC180', 190: 'SC190'}, 'MoSold': {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}})
train = train.replace({'Alley': {'Grvl': 1, 'Pave': 2}, 'BsmtCond': {'No': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'BsmtExposure': {'No': 0, 'Mn': 1, 'Av': 2, 'Gd': 3}, 'BsmtFinType1': {'No': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}, 'BsmtFinType2': {'No': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}, 'BsmtQual': {'No': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'ExterCond': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'ExterQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'FireplaceQu': {'No': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'Functional': {'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8}, 'GarageCond': {'No': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'GarageQual': {'No': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'HeatingQC': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'KitchenQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'LandSlope': {'Sev': 1, 'Mod': 2, 'Gtl': 3}, 'LotShape': {'IR3': 1, 'IR2': 2, 'IR1': 3, 'Reg': 4}, 'PavedDrive': {'N': 0, 'P': 1, 'Y': 2}, 'PoolQC': {'No': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}, 'Street': {'Grvl': 1, 'Pave': 2}, 'Utilities': {'ELO': 1, 'NoSeWa': 2, 'NoSewr': 3, 'AllPub': 4}})
train['SimplOverallQual'] = train.OverallQual.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 3})
train['SimplOverallCond'] = train.OverallCond.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 3})
train['SimplPoolQC'] = train.PoolQC.replace({1: 1, 2: 1, 3: 2, 4: 2})
train['SimplGarageCond'] = train.GarageCond.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
train['SimplGarageQual'] = train.GarageQual.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
train['SimplFireplaceQu'] = train.FireplaceQu.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
train['SimplFireplaceQu'] = train.FireplaceQu.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
train['SimplFunctional'] = train.Functional.replace({1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 3, 8: 4})
train['SimplKitchenQual'] = train.KitchenQual.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
train['SimplHeatingQC'] = train.HeatingQC.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
train['SimplBsmtFinType1'] = train.BsmtFinType1.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2})
train['SimplBsmtFinType2'] = train.BsmtFinType2.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2})
train['SimplBsmtCond'] = train.BsmtCond.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
train['SimplBsmtQual'] = train.BsmtQual.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
train['SimplExterCond'] = train.ExterCond.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
train['SimplExterQual'] = train.ExterQual.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
train['OverallGrade'] = train['OverallQual'] * train['OverallCond']
train['GarageGrade'] = train['GarageQual'] * train['GarageCond']
train['ExterGrade'] = train['ExterQual'] * train['ExterCond']
train['KitchenScore'] = train['KitchenAbvGr'] * train['KitchenQual']
train['FireplaceScore'] = train['Fireplaces'] * train['FireplaceQu']
train['GarageScore'] = train['GarageArea'] * train['GarageQual']
train['PoolScore'] = train['PoolArea'] * train['PoolQC']
train['SimplOverallGrade'] = train['SimplOverallQual'] * train['SimplOverallCond']
train['SimplExterGrade'] = train['SimplExterQual'] * train['SimplExterCond']
train['SimplPoolScore'] = train['PoolArea'] * train['SimplPoolQC']
train['SimplGarageScore'] = train['GarageArea'] * train['SimplGarageQual']
train['SimplFireplaceScore'] = train['Fireplaces'] * train['SimplFireplaceQu']
train['SimplKitchenScore'] = train['KitchenAbvGr'] * train['SimplKitchenQual']
train['TotalBath'] = train['BsmtFullBath'] + 0.5 * train['BsmtHalfBath'] + train['FullBath'] + 0.5 * train['HalfBath']
train['AllSF'] = train['GrLivArea'] + train['TotalBsmtSF']
train['AllFlrsSF'] = train['1stFlrSF'] + train['2ndFlrSF']
train['AllPorchSF'] = train['OpenPorchSF'] + train['EnclosedPorch'] + train['3SsnPorch'] + train['ScreenPorch']
train['HasMasVnr'] = train.MasVnrType.replace({'BrkCmn': 1, 'BrkFace': 1, 'CBlock': 1, 'Stone': 1, 'None': 0})
train['BoughtOffPlan'] = train.SaleCondition.replace({'Abnorml': 0, 'Alloca': 0, 'AdjLand': 0, 'Family': 0, 'Normal': 0, 'Partial': 1})
print('Find most important features relative to target')
corr = train.corr()
corr.sort_values(['SalePrice'], ascending=False, inplace=True)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(corr.SalePrice)
train['OverallQual-s2'] = train['OverallQual'] ** 2
train['OverallQual-s3'] = train['OverallQual'] ** 3
train['OverallQual-Sq'] = np.sqrt(train['OverallQual'])
train['AllSF-2'] = train['AllSF'] ** 2
train['AllSF-3'] = train['AllSF'] ** 3
train['AllSF-Sq'] = np.sqrt(train['AllSF'])
train['AllFlrsSF-2'] = train['AllFlrsSF'] ** 2
train['AllFlrsSF-3'] = train['AllFlrsSF'] ** 3
train['AllFlrsSF-Sq'] = np.sqrt(train['AllFlrsSF'])
train['GrLivArea-2'] = train['GrLivArea'] ** 2
train['GrLivArea-3'] = train['GrLivArea'] ** 3
train['GrLivArea-Sq'] = np.sqrt(train['GrLivArea'])
train['SimplOverallQual-s2'] = train['SimplOverallQual'] ** 2
train['SimplOverallQual-s3'] = train['SimplOverallQual'] ** 3
train['SimplOverallQual-Sq'] = np.sqrt(train['SimplOverallQual'])
train['ExterQual-2'] = train['ExterQual'] ** 2
train['ExterQual-3'] = train['ExterQual'] ** 3
train['ExterQual-Sq'] = np.sqrt(train['ExterQual'])
train['GarageCars-2'] = train['GarageCars'] ** 2
train['GarageCars-3'] = train['GarageCars'] ** 3
train['GarageCars-Sq'] = np.sqrt(train['GarageCars'])
train['TotalBath-2'] = train['TotalBath'] ** 2
train['TotalBath-3'] = train['TotalBath'] ** 3
train['TotalBath-Sq'] = np.sqrt(train['TotalBath'])
train['KitchenQual-2'] = train['KitchenQual'] ** 2
train['KitchenQual-3'] = train['KitchenQual'] ** 3
train['KitchenQual-Sq'] = np.sqrt(train['KitchenQual'])
train['GarageScore-2'] = train['GarageScore'] ** 2
train['GarageScore-3'] = train['GarageScore'] ** 3
train['GarageScore-Sq'] = np.sqrt(train['GarageScore'])
cat_features = train.select_dtypes(include=['object']).columns
num_features = train.select_dtypes(exclude=['object']).columns
num_features = num_features.drop('SalePrice')
train_num = train[num_features]
train_cat = train[cat_features]
print('before', train_num.isnull().values.sum())
train_num = train_num.fillna(train_num.median())
print('after', train_num.isnull().values.sum())
from scipy.stats import skew
skewness = train_num.apply(lambda x: skew(x))
skewness = skewness[abs(skewness) > 0.5]
skewed_features = skewness.index
train_num[skewed_features] = np.log1p(train_num[skewed_features])
train_cat = pd.get_dummies(train_cat)
print(train)
from sklearn.model_selection import train_test_split
train = pd.concat([train_num, train_cat], axis=1)
(X_train, X_test, y_train, y_test) = train_test_split(train, y, test_size=0.3, random_state=42)
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
train.loc[:, num_features] = stdSc.fit_transform(train.loc[:, num_features])
rmse_list = []
models = [lasso, ridge, elasticnet, gbr, lightgbm, xgboost]