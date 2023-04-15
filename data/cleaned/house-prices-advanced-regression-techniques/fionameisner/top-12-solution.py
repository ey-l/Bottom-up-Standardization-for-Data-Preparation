from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train.head(10)
X = pd.DataFrame(train.loc[:, train.columns != 'SalePrice'])
X = pd.concat([X, test], axis=0)
y = pd.DataFrame(train['SalePrice'])
X.info()
(fig, ax) = plt.subplots(1, 2)
ax[0].hist(train['SalePrice'], bins=12, edgecolor='green', facecolor='yellow')
ax[0].set_title('SalePrice distribution ')
ax[1].hist(np.log(y['SalePrice']), bins=12, edgecolor='white')
ax[1].set_title('SalePrice Log')
plt.subplots_adjust(right=0.9, wspace=0.4, hspace=0.4)
plt.figure(figsize=(12, 9))
sns.heatmap(X.isnull(), cmap='YlGnBu')
X.isnull().sum().sum()
X['Functional'].fillna('Typ', inplace=True)
X['Electrical'].fillna('SBrkr', inplace=True)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'MiscFeature', 'Fence', 'FireplaceQu', 'Alley', 'PoolQC'):
    X[col] = X[col].fillna('No')
for col in ('GarageArea', 'GarageCars'):
    X[col] = X[col].fillna(0)
for col in ('MSZoning', 'Utilities', 'MasVnrType', 'Exterior1st', 'Exterior2nd', 'SaleType'):
    X[col] = X[col].fillna(X[col].mode()[0])
X.fillna(X.median(), inplace=True)
X['AllSF'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']
X['BackyardSF'] = X['LotArea'] - X['1stFlrSF']
X['PorchSF'] = X['WoodDeckSF'] + X['OpenPorchSF'] + X['EnclosedPorch'] + X['3SsnPorch'] + X['ScreenPorch']
X['Total_Bathrooms'] = X['FullBath'] + X['BsmtFullBath'] + 0.5 * X['HalfBath'] + 0.5 * X['BsmtHalfBath']
X['MedNhbdArea'] = X.groupby('Neighborhood')['GrLivArea'].transform('median')
X['IsAbvGr'] = X[['MedNhbdArea', 'GrLivArea']].apply(lambda x: 'yes' if x['GrLivArea'] > x['MedNhbdArea'] else 'no', axis=1)
corrmat = X.corr()
(f, ax) = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)
scale = ['MedNhbdArea', 'BackyardSF', 'PorchSF', 'WoodDeckSF', 'OpenPorchSF', 'AllSF', '1stFlrSF', '2ndFlrSF', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'GarageArea', 'GrLivArea', 'LotArea', 'LotFrontage', 'LowQualFinSF', 'MasVnrArea', 'TotalBsmtSF', 'PoolArea']
encode = list(set(X.columns) - set(scale) - set(['Id']))
skew_feats = X[scale].skew().sort_values(ascending=False)
skewness = pd.DataFrame({'Skew': skew_feats.astype('float')})
skewness = skewness[skewness.Skew > 0.75]
indeces = list(skewness.index)
for x in indeces:
    X[x] = np.log1p(X[x])
Xscale = X[scale]