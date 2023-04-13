import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_rows = None
pd.options.display.max_columns = None
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1.shape
_input1.info()
_input1.head()
_input1.describe().transpose()
_input1.hist(figsize=(20, 20), bins=20)
plt.figure(figsize=(26, 16))
sns.heatmap(_input1.corr(), cmap='rocket', annot=True, fmt=f'0.1', cbar=False)
plt.figure(figsize=(12, 4))
sns.distplot(_input1['SalePrice'])
plt.figure(figsize=(12, 4))
sns.distplot(np.log(_input1['SalePrice']))
_input1.shape
_input1['LogPrice'] = np.log(_input1['SalePrice'])
_input1 = _input1.drop('SalePrice', axis=1)
_input1.corr()['LogPrice'].sort_values(ascending=False)
sns.barplot(x='OverallQual', y='LogPrice', data=_input1)
sns.scatterplot(x='GrLivArea', y='LogPrice', data=_input1)
sns.scatterplot(x='GarageArea', y='LogPrice', data=_input1)
sns.scatterplot(x='TotalBsmtSF', y='LogPrice', data=_input1)
sns.scatterplot(x='LotFrontage', y='LogPrice', data=_input1)
sns.scatterplot(x='LotArea', y='LogPrice', data=_input1)
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
test_id = _input0.pop('Id')
train_id = _input1.pop('Id')
n_train = _input1.shape[0]
labels = _input1.pop('LogPrice')
df = pd.concat([_input1, _input0], axis=0)
df = df.reset_index(inplace=False, drop=True)
df.shape
(_input0.shape, _input1.shape)
pd.DataFrame({'Amount': df.isnull().sum(), 'Percent': df.isnull().sum() / len(df) * 100}).sort_values(by='Percent', ascending=False)
df = df.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1)
df['Fireplaces'].value_counts()
df['FireplaceQu'].value_counts()
df['FireplaceQu'] = df['FireplaceQu'].fillna('NA', inplace=False)
sns.distplot(df['LotFrontage'])
lot_frontage_median = df['LotFrontage'].median()
df['LotFrontage'] = df['LotFrontage'].fillna(lot_frontage_median)
df[df['GarageYrBlt'].isnull()].head()
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
for column in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    df[column] = df[column].fillna('NA')
df[df['BsmtExposure'].isnull()].head()
for column in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    df[column] = df[column].fillna('NA')
df[df['MasVnrType'].isnull()].head()
df['MasVnrType'].value_counts()
df['MasVnrType'] = df['MasVnrType'].fillna('None')
df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
df['MSZoning'].value_counts()
df['MSZoning'] = df['MSZoning'].fillna(df['MSZoning'].mode()[0])
df['BsmtHalfBath'].value_counts()
df['BsmtFullBath'].value_counts()
df['BsmtFullBath'] = df['BsmtFullBath'].fillna(0)
df['BsmtHalfBath'] = df['BsmtHalfBath'].fillna(0)
df['Functional'].value_counts()
df['Functional'] = df['Functional'].fillna(df['Functional'].mode()[0])
df['Electrical'].value_counts()
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
df['Utilities'].value_counts()
df['Utilities'] = df['Utilities'].fillna(df['Utilities'].mode()[0])
df[df['TotalBsmtSF'].isnull()]
df.head(5)
df['TotalBsmtSF'] = df['TotalBsmtSF'].fillna(df['1stFlrSF'])
df['BsmtUnfSF'] = df['BsmtUnfSF'].fillna(0)
df['BsmtFinSF2'] = df['BsmtFinSF2'].fillna(0)
df['BsmtFinSF1'] = df['BsmtFinSF1'].fillna(0)
df[df['GarageCars'].isnull()]
df['GarageCars'] = df['GarageCars'].fillna(0)
df['GarageArea'] = df['GarageArea'].fillna(0)
df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])
df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])
df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])
df['KitchenQual'] = df['KitchenQual'].fillna(df['KitchenQual'].mode()[0])
pd.DataFrame({'Amount': df.isnull().sum(), 'Percent': df.isnull().sum() / len(df) * 100}).sort_values(by='Percent', ascending=False)
qual_columns = ['GarageCond', 'GarageQual', 'FireplaceQu', 'KitchenQual', 'HeatingQC', 'BsmtCond', 'BsmtQual', 'ExterCond', 'ExterQual']
bsmt_columns = ['BsmtFinType2', 'BsmtFinType1']
exposure_columns = ['BsmtExposure']
qual_rates = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
bsmtype_rates = {'GLQ': 5, 'ALQ': 4, 'BLQ': 3, 'Rec': 2, 'LwQ': 1, 'Unf': -1, 'NA': 0}
exposure_rates = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': -1, 'NA': 0}
for (feats, rate) in ((qual_columns, qual_rates), (bsmt_columns, bsmtype_rates), (exposure_columns, exposure_rates)):
    for feat in feats:
        df[feat] = df[feat].map(rate)
encode = ['Functional', 'CentralAir', 'PavedDrive', 'GarageFinish', 'Street', 'LandSlope']
for feat in encode:
    df['{0}_cat'.format(feat)] = pd.factorize(df[feat])[0]
categorical_features = [x for x in df.select_dtypes(include=np.object).columns if x not in encode]
for feat in categorical_features:
    dummies = pd.get_dummies(df[feat], prefix='{0}'.format(feat), drop_first=True)
    df = pd.concat([df, dummies], axis=1)
df['BsmtFin'] = df['BsmtFinSF1'] + df['BsmtFinSF2']
df['TotalBsmt'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
df.shape
train_set = df[:n_train]
test_set = df[n_train:]
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor, VotingRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
X = train_set[train_set.select_dtypes(include=np.number).columns].values
scalar = MinMaxScaler()
X_scaled = scalar.fit_transform(X)
lasso_model = Lasso()
elastic_model = ElasticNet()
svr_model = SVR()
tree_model = ExtraTreesRegressor()
xgb_model = XGBRegressor()
knn_model = KNeighborsRegressor()
models = {'lasso_model': lasso_model, 'elastic_model': elastic_model, 'svr_model': svr_model, 'tree_model': tree_model, 'xgb_model': xgb_model, 'knn_model': knn_model}

def cross_validation(model, X, y):
    """Check model with cross validation"""
    score = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    cross_score = np.sqrt(-score)
    return round(np.mean(cross_score), 4)
models_evaluation = {}
for (model_name, model) in models.items():
    models_evaluation[model_name] = cross_validation(model, X_scaled, labels)
pd.DataFrame(data=models_evaluation.items(), columns=['Model', 'RMSE']).sort_values(by='RMSE')