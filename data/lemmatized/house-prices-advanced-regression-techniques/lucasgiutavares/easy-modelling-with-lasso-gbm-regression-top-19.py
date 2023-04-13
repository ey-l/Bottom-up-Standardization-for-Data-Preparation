import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.svm import SVC as svc
import sklearn.linear_model as sk
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.stats as sp
from scipy.special import boxcox1p
from sklearn.feature_selection import RFECV
import sklearn.metrics as mt
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
no_features = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageYrBlt', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']
_input1[no_features] = _input1[no_features].fillna('No')
_input1.loc[:, 'LotFrontage_NA'] = _input1.LotFrontage.isnull() * 1
_input1.loc[:, 'LotFrontage_NA'] = _input1.loc[:, 'LotFrontage_NA'].astype('object')

def HasFeat(var):
    hasf = var
    hasf[hasf != 'No'] = 1
    hasf[hasf == 'No'] = 0
    return hasf

def feat_eng(tr):
    tr.GarageQual[tr['GarageQual'] == 'No'] = 0
    tr.GarageQual[tr['GarageQual'] == 'Po'] = 1
    tr.GarageQual[tr['GarageQual'] == 'Fa'] = 2
    tr.GarageQual[tr['GarageQual'] == 'TA'] = 3
    tr.GarageQual[tr['GarageQual'] == 'Gd'] = 4
    tr.GarageQual[tr['GarageQual'] == 'Ex'] = 5
    tr.GarageQual = tr.GarageQual.astype('float32')
    tr.GarageCond[tr['GarageCond'] == 'No'] = 0
    tr.GarageCond[tr['GarageCond'] == 'Po'] = 1
    tr.GarageCond[tr['GarageCond'] == 'Fa'] = 2
    tr.GarageCond[tr['GarageCond'] == 'TA'] = 3
    tr.GarageCond[tr['GarageCond'] == 'Gd'] = 4
    tr.GarageCond[tr['GarageCond'] == 'Ex'] = 5
    tr.GarageCond = tr.GarageCond.astype('float32')
    tr.GarageYrBlt[tr['GarageYrBlt'] == 'No'] = 0
    tr.GarageYrBlt = tr.GarageYrBlt.astype('float32')
    tr.Functional[tr['Functional'] == 'Typ'] = 7
    tr.Functional[tr['Functional'] == 'Min1'] = 6
    tr.Functional[tr['Functional'] == 'Min2'] = 5
    tr.Functional[tr['Functional'] == 'Mod'] = 4
    tr.Functional[tr['Functional'] == 'Maj1'] = 3
    tr.Functional[tr['Functional'] == 'Maj2'] = 2
    tr.Functional[tr['Functional'] == 'Sev'] = 1
    tr.Functional[tr['Functional'] == 'Sal'] = 0
    tr.Functional = tr.Functional.astype('float32')
    tr.Fence[tr['Fence'] == 'No'] = 0
    tr.Fence[tr['Fence'] == 'MnWw'] = 1
    tr.Fence[tr['Fence'] == 'GdWo'] = 2
    tr.Fence[tr['Fence'] == 'MnPrv'] = 3
    tr.Fence[tr['Fence'] == 'GdPrv'] = 4
    tr.Fence = tr.Fence.astype('float32')
    tr.KitchenQual[tr['KitchenQual'] == 'Po'] = 1
    tr.KitchenQual[tr['KitchenQual'] == 'Fa'] = 2
    tr.KitchenQual[tr['KitchenQual'] == 'TA'] = 3
    tr.KitchenQual[tr['KitchenQual'] == 'Gd'] = 4
    tr.KitchenQual[tr['KitchenQual'] == 'Ex'] = 5
    tr.KitchenQual = tr.KitchenQual.astype('float32')
    tr.HeatingQC[tr['HeatingQC'] == 'Po'] = 1
    tr.HeatingQC[tr['HeatingQC'] == 'Fa'] = 2
    tr.HeatingQC[tr['HeatingQC'] == 'TA'] = 3
    tr.HeatingQC[tr['HeatingQC'] == 'Gd'] = 4
    tr.HeatingQC[tr['HeatingQC'] == 'Ex'] = 5
    tr.HeatingQC = tr.HeatingQC.astype('float32')
    tr.ExterQual[tr['ExterQual'] == 'Po'] = 1
    tr.ExterQual[tr['ExterQual'] == 'Fa'] = 2
    tr.ExterQual[tr['ExterQual'] == 'TA'] = 3
    tr.ExterQual[tr['ExterQual'] == 'Gd'] = 4
    tr.ExterQual[tr['ExterQual'] == 'Ex'] = 5
    tr.ExterQual = tr.ExterQual.astype('float32')
    tr.BsmtQual[tr['BsmtQual'] == 'No'] = 0
    tr.BsmtQual[tr['BsmtQual'] == 'Po'] = 1
    tr.BsmtQual[tr['BsmtQual'] == 'Fa'] = 2
    tr.BsmtQual[tr['BsmtQual'] == 'TA'] = 3
    tr.BsmtQual[tr['BsmtQual'] == 'Gd'] = 4
    tr.BsmtQual[tr['BsmtQual'] == 'Ex'] = 5
    tr.BsmtQual = tr.BsmtQual.astype('float32')
    tr.BsmtFinType1[tr['BsmtFinType1'] == 'No'] = 0
    tr.BsmtFinType1[tr['BsmtFinType1'] == 'Unf'] = 1
    tr.BsmtFinType1[tr['BsmtFinType1'] == 'LwQ'] = 2
    tr.BsmtFinType1[tr['BsmtFinType1'] == 'Rec'] = 3
    tr.BsmtFinType1[tr['BsmtFinType1'] == 'BLQ'] = 4
    tr.BsmtFinType1[tr['BsmtFinType1'] == 'ALQ'] = 5
    tr.BsmtFinType1[tr['BsmtFinType1'] == 'GLQ'] = 6
    tr.BsmtFinType1 = tr.BsmtFinType1.astype('float32')
    tr.BsmtFinType2[tr['BsmtFinType2'] == 'No'] = 0
    tr.BsmtFinType2[tr['BsmtFinType2'] == 'Unf'] = 1
    tr.BsmtFinType2[tr['BsmtFinType2'] == 'LwQ'] = 2
    tr.BsmtFinType2[tr['BsmtFinType2'] == 'Rec'] = 3
    tr.BsmtFinType2[tr['BsmtFinType2'] == 'BLQ'] = 4
    tr.BsmtFinType2[tr['BsmtFinType2'] == 'ALQ'] = 5
    tr.BsmtFinType2[tr['BsmtFinType2'] == 'GLQ'] = 6
    tr.BsmtFinType2 = tr.BsmtFinType2.astype('float32')
    tr['TotalArea'] = tr['TotalBsmtSF'] + tr['1stFlrSF'] + tr['2ndFlrSF']
    tr['BsmtFinArea'] = (tr['BsmtFinSF1'] + tr['BsmtFinSF2']) / (tr['BsmtFinSF1'] + tr['BsmtFinSF2'] + tr['BsmtUnfSF'])
    tr['MSSubClass'] = tr['MSSubClass'].astype('object')
    return tr
_input1 = feat_eng(_input1)
plt.scatter(_input1['GrLivArea'], _input1['SalePrice'])
plt.scatter([4676, 5642], [184750, 160000], color='red')
_input1 = _input1.drop(_input1[(_input1['GrLivArea'] > 4000) & (_input1['SalePrice'] < 300000)].index)
plt.scatter(_input1['GrLivArea'], _input1['SalePrice'])
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
num = _input1.select_dtypes(include=numerics)
skew = pd.DataFrame(sp.skew(num), num.keys())
skew = skew.sort_values(by=0, ascending=False)
plt.figure(figsize=(15, 7))
plt.plot(skew)
plt.xticks(rotation=45, fontsize=8)
plt.xlabel('Numerical Features', fontsize=10)
plt.ylabel('Skew', fontsize=10)
plt.axhline(y=1, c='red')
plt.axhline(y=-1, c='red')

def boxcox(a, f):
    if sp.skew(a) not in range(-1, 1) or sp.kurtosis(a) not in range(-1, 1):
        a = boxcox1p(a, 0.05)
        print(f, 'transformed.')
        return a
transformed_feats = []
for f in _input1.keys():
    if _input1[f].dtype == object:
        _input1[f].fillna(value=_input1[f].value_counts().idxmax())
        _input1[f] = _input1[f].astype('category')
    else:
        _input1[f] = _input1[f].fillna(value=np.mean(_input1[f]))
        if f != 'SalePrice':
            _input1[f] = boxcox(_input1[f], f)
            _input1[f + 'Sq'] = _input1[f] ** 2
            _input1[f + 'Cub'] = _input1[f] ** 3
            transformed_feats.append(f)
num = _input1[skew.index].select_dtypes(include=numerics)
skew = pd.DataFrame(sp.skew(num), num.keys())
skew = skew.sort_values(by=0, ascending=False)
plt.figure(figsize=(15, 7))
plt.plot(skew)
plt.xticks(rotation=45, fontsize=8)
plt.xlabel('Numerical Features', fontsize=10)
plt.ylabel('Skew', fontsize=10)
plt.axhline(y=1, c='red')
plt.axhline(y=-1, c='red')
cols = _input1.columns
_input1 = pd.get_dummies(_input1, drop_first=True)
all_features = _input1.keys()
_input1 = _input1.drop(_input1.loc[:, (_input1 == 0).sum() >= 1444], axis=1)
_input1 = _input1.drop(_input1.loc[:, (_input1 == 1).sum() >= 1444], axis=1)
remain_features = _input1.keys()
remov_features = [st for st in all_features if st not in remain_features]
print(len(remov_features), 'features were removed:', remov_features)
plt.hist(_input1['SalePrice'])
sp.probplot(_input1['SalePrice'], plot=plt)
y = np.log(_input1['SalePrice'].values)
plt.hist(y)
sp.probplot(y, plot=plt)
X = _input1.drop('SalePrice', axis=1)

def scorer(estimator, X, y):
    y_new = estimator.predict(X)
    return np.sqrt(mt.mean_squared_error(y, y_new))
est = sk.ElasticNet(l1_ratio=0, alpha=0.017, random_state=1)
fsel = RFECV(est, step=1, cv=15, n_jobs=-1, scoring=scorer)