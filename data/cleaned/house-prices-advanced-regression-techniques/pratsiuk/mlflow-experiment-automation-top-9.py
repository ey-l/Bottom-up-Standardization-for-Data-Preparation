import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import optuna
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from scipy.stats import norm, skew, boxcox_normmax
from sklearn import utils
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, MinMaxScaler, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE, f_regression
from scipy.special import boxcox1p
import datetime
import time
from mlflow import log_metric, log_param, log_artifact
import mlflow.sklearn
import mlflow.xgboost
experiment_name = 'house_price'
mlflow.set_experiment(experiment_name)
mlflow.end_run()
artifact_path = mlflow.get_artifact_uri()
uri = mlflow.tracking.get_tracking_uri()
print(artifact_path)
print(uri)

def log_mlflow(model):
    with mlflow.start_run() as run:
        mlflow.set_tag('model_name', name)
        mlflow.log_param('CV_n_folds', CV_n_folds)
        mlflow.log_param('TEST_PART', TEST_PART)
        mlflow.log_param('Train size', X_train.shape)
        mlflow.log_param('Colums', str(X_train.columns.values.tolist()))
        mlflow.log_metrics({'rmse_cv': score_cv.mean(), 'rmse': score})
        mlflow.log_metric('rmse_train', score_train)
        mlflow.sklearn.log_model(model, name)
    mlflow.end_run()
mlflow.end_run()
path = '_data/input/house-prices-advanced-regression-techniques/'
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
print('Size ntrain= {} / ntest = {}'.format(train.shape, test.shape))
train['SalePrice'] = np.log1p(train['SalePrice'])
train = train.drop(train[train['GrLivArea'] > 4600].index)
train = train.drop(train[train['TotalBsmtSF'] > 5900].index)
train = train.drop(train[train['1stFlrSF'] > 4000].index)
train = train.drop(train[train['MasVnrArea'] > 1500].index)
train = train.drop(train[train['GarageArea'] > 1230].index)
train = train.drop(train[train['TotRmsAbvGrd'] > 13].index)
ntrain = train.shape[0]
ntest = test.shape[0]
print('Size ntrain= {} / ntest = {}'.format(ntrain, ntest))
all_data = train.append(test, sort=False).reset_index(drop=True)
orig_test = test.copy()
log_y_train = train['SalePrice']
all_data.drop(['SalePrice'], axis=1, inplace=True)
all_data.drop(['Id'], axis=1, inplace=True)
print('all_data size is : {}'.format(all_data.shape))
all_data_na = all_data.isnull().sum() / len(all_data) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio': all_data_na})
missing_data.head(20)
threshold = 0.9
corr_matrix = train.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
collinear_features = [column for column in upper.columns if any(upper[column] > threshold)]
print('There are %d features to remove.' % len(collinear_features))
print(collinear_features)
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
all_data['PoolQC'] = all_data['PoolQC'].fillna('None')
all_data['MiscFeature'] = all_data['MiscFeature'].fillna('None')
all_data['Alley'] = all_data['Alley'].fillna('None')
all_data['Fence'] = all_data['Fence'].fillna('None')
all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna('None')
all_data['MasVnrType'] = all_data['MasVnrType'].fillna('None')
all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data['Functional'] = all_data['Functional'].fillna('Typ')
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['MSSubClass'] = all_data['MSSubClass'].fillna('None')
all_data = all_data.drop(['Utilities'], axis=1)
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

def convert_str_to_int(data, features, score):
    all_data[features] = all_data[features].applymap(lambda s: score.get(s) if s in score else s)
featuresQualCond = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'KitchenQual', 'FireplaceQu', 'HeatingQC', 'GarageQual', 'GarageCond', 'PoolQC']
qual_score_QualCond = {'None': 0, 'NA': 1, 'Po': 2, 'Fa': 3, 'TA': 4, 'Gd': 5, 'Ex': 6}
convert_str_to_int(all_data, featuresQualCond, qual_score_QualCond)
featuresExposure = ['BsmtExposure']
qual_score = {'None': 0, 'NA': 1, 'No': 2, 'Mn': 3, 'Av': 4, 'Gd': 5}
convert_str_to_int(all_data, featuresExposure, qual_score)
featuresFinType = ['BsmtFinType1', 'BsmtFinType2']
qual_score = {'None': 0, 'NA': 1, 'Unf': 2, 'LwQ': 3, 'Rec': 4, 'BLQ': 5, 'ALQ': 6, 'GLQ': 7}
convert_str_to_int(all_data, featuresFinType, qual_score)
featuresGarageFin = ['GarageFinish']
qual_score = {'None': 0, 'NA': 1, 'Unf': 2, 'RFn': 3, 'Fin': 4}
convert_str_to_int(all_data, featuresGarageFin, qual_score)
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
all_data['OverallCond'] = all_data['OverallCond'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope', 'LotShape', 'PavedDrive', 'CentralAir', 'MSSubClass', 'OverallCond', 'YrSold', 'MoSold')
for c in cols:
    lbl = LabelEncoder()