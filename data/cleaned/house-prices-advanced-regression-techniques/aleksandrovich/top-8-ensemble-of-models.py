import sys
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.regressor import StackingCVRegressor
from scipy.stats import skew
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
print('Imports have been set')
if not sys.warnoptions:
    warnings.simplefilter('ignore')
X = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
X_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
rows_before = X.shape[0]
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
rows_after = X.shape[0]
print('\nRows containing NaN in SalePrice were dropped: ' + str(rows_before - rows_after))
X['SalePrice'] = np.log1p(X['SalePrice'])
y = X['SalePrice'].reset_index(drop=True)
train_features = X.drop(['SalePrice'], axis=1)
features = pd.concat([train_features, X_test]).reset_index(drop=True)
print('\nFeatures size:', features.shape)
nan_count_table = features.isnull().sum()
nan_count_table = nan_count_table[nan_count_table > 0].sort_values(ascending=False)
print('\nColums containig NaN: ')
print(nan_count_table)
columns_containig_nan = nan_count_table.index.to_list()
print('\nWhat values they contain: ')
print(features[columns_containig_nan])
for column in columns_containig_nan:
    if column in ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'TotalBsmtSF', 'Fireplaces', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea']:
        features[column] = features[column].fillna(0)
    if column in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Neighborhood', 'BldgType', 'HouseStyle', 'MasVnrType', 'FireplaceQu', 'Fence', 'MiscFeature']:
        features[column] = features[column].fillna('None')
    if column in ['Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Condition1', 'Condition2', 'RoofStyle', 'Electrical', 'Functional', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'RoofMatl', 'ExterQual', 'ExterCond', 'Foundation', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'PavedDrive', 'SaleType', 'SaleCondition']:
        features[column] = features[column].fillna(features[column].mode()[0])
features['MSSubClass'] = features['MSSubClass'].apply(str)
features['MSSubClass'] = features['MSSubClass'].fillna('Unknown')
features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
features['LotArea'] = features['LotArea'].astype(np.int64)
features['Alley'] = features['Alley'].fillna('Pave')
features['MasVnrArea'] = features['MasVnrArea'].astype(np.int64)
features['YrBltAndRemod'] = features['YearBuilt'] + features['YearRemodAdd']
features['TotalSF'] = features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']
features['Total_sqr_footage'] = features['BsmtFinSF1'] + features['BsmtFinSF2'] + features['1stFlrSF'] + features['2ndFlrSF']
features['Total_Bathrooms'] = features['FullBath'] + 0.5 * features['HalfBath'] + features['BsmtFullBath'] + 0.5 * features['BsmtHalfBath']
features['Total_porch_sf'] = features['OpenPorchSF'] + features['3SsnPorch'] + features['EnclosedPorch'] + features['ScreenPorch'] + features['WoodDeckSF']
features['haspool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
features['has2ndfloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
features['hasgarage'] = features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
features['hasbsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
features['hasfireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
print('Features size:', features.shape)
nan_count_train_table = features.isnull().sum()
nan_count_train_table = nan_count_train_table[nan_count_train_table > 0].sort_values(ascending=False)
print('\nAre no NaN here now: ' + str(nan_count_train_table.size == 0))
numeric_columns = [cname for cname in features.columns if features[cname].dtype in ['int64', 'float64']]
print('\nColumns which are numeric: ' + str(len(numeric_columns)) + ' out of ' + str(features.shape[1]))
print(numeric_columns)
categoric_columns = [cname for cname in features.columns if features[cname].dtype == 'object']
print('\nColumns whice are categoric: ' + str(len(categoric_columns)) + ' out of ' + str(features.shape[1]))
print(categoric_columns)
skewness = features[numeric_columns].apply(lambda x: skew(x))
print(skewness.sort_values(ascending=False))
skewness = skewness[abs(skewness) > 0.7]
print('\nSkewed values: ' + str(skewness.index))
fixed_features = np.log1p(features[skewness.index])
print("\nLet's see and compare skewed features")
for column in skewness.index[0:5]:
    sns.distplot(features[column])

    sns.distplot(fixed_features[column])

    print('---------------------------------------------------')
print('\nSkewed values: ' + str(skewness.index))
features[skewness.index] = np.log1p(features[skewness.index])
final_features = pd.get_dummies(features).reset_index(drop=True)
X = final_features.iloc[:len(y), :]
X_test = final_features.iloc[len(X):, :]
print('Features size for train(X,y) and test(X_test):')
print('X', X.shape, 'y', y.shape, 'X_test', X_test.shape)
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]
alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))
lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=10000000.0, alphas=alphas2, random_state=14, cv=kfolds))
elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=10000000.0, alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))
gboost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state=5)
lgbm = lgb.LGBMRegressor(objective='regression', num_leaves=5, learning_rate=0.05, n_estimators=720, max_bin=55, bagging_fraction=0.8, bagging_freq=5, feature_fraction=0.2319, feature_fraction_seed=9, bagging_seed=9, min_data_in_leaf=6, min_sum_hessian_in_leaf=11)
c_grid = {'n_estimators': [1000], 'early_stopping_rounds': [1], 'learning_rate': [0.1]}
xgb_regressor = XGBRegressor(objective='reg:squarederror')
cross_validation = KFold(n_splits=10, shuffle=True, random_state=2)
xgb_r = GridSearchCV(estimator=xgb_regressor, param_grid=c_grid, cv=cross_validation)
stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, lgbm, gboost), meta_regressor=elasticnet, use_features_in_secondary=True)
svr = make_pipeline(RobustScaler(), SVR(C=20, epsilon=0.008, gamma=0.0003))
print('\n\nFitting our models ensemble: ')
print('Elasticnet is fitting now...')