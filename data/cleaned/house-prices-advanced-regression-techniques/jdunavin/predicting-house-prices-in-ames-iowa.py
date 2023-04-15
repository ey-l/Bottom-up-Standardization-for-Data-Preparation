import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import matplotlib.ticker as tick
import seaborn as sb
from sklearn import linear_model
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', header=0)
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', header=0)
train = train.fillna(0)
test = test.fillna(0)
features = [x for x in train.columns if x not in ['id', 'SalePrice']]
cat_features = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'Foundation', 'BsmtFinType1', 'BsmtFinType2', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Heating', 'CentralAir', 'Electrical', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'GarageType', 'GarageFinish', 'GarageCars', 'PavedDrive', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
ord_cat_features = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
num_features = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'LowQualFinSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'YearRemodAdd', 'YearBuilt', 'GarageYrBlt', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']
rows_train = len(train)
rows_test = len(test)

def transform_nb(x):
    if x in ('NoRidge', 'NridgHt', 'StoneBr', 'Timber', 'Veenker'):
        return 'Hi'
    elif x in ('Mitchel', 'OldTown', 'BrkSide', 'Sawyer', 'NAmes', 'IDOTRR', 'MeadowV', 'Edwards', 'NPkVill', 'BrDale', 'Blueste'):
        return 'Low'
    else:
        return 'Mid'
train_test = pd.concat((train[features], test[features])).reset_index(drop=True)
for c in range(len(cat_features)):
    train_test[cat_features[c]] = train_test[cat_features[c]].astype('category').cat.codes

def coding(col, codeDict):
    colCoded = pd.Series(col, copy=True)
    for (key, value) in codeDict.items():
        colCoded.replace(key, value, inplace=True)
    return colCoded
train_test['ExterQual'] = coding(train_test['ExterQual'], {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
train_test['ExterCond'] = coding(train_test['ExterCond'], {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
train_test['BsmtQual'] = coding(train_test['BsmtQual'], {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 0: 0})
train_test['BsmtCond'] = coding(train_test['BsmtCond'], {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 0: 0})
train_test['BsmtExposure'] = coding(train_test['BsmtExposure'], {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 0: 0})
train_test['HeatingQC'] = coding(train_test['HeatingQC'], {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
train_test['KitchenQual'] = coding(train_test['KitchenQual'], {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
train_test['FireplaceQu'] = coding(train_test['FireplaceQu'], {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 0: 0})
train_test['GarageQual'] = coding(train_test['GarageQual'], {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 0: 0})
train_test['GarageCond'] = coding(train_test['GarageCond'], {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 0: 0})
train_test['PoolQC'] = coding(train_test['PoolQC'], {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 0: 0})
trainX = train_test.iloc[:rows_train, :]
testX = train_test.iloc[rows_train:, :]
fig = plt.figure()
ax = plt.axes()
(n, bins, patches) = plt.hist(train['SalePrice'], 30, facecolor='dimgrey')
plt.xlabel('Sale price (000s)')
vals = ax.get_xticks()
ax.set_xticklabels(['${:,.0f}'.format(x / 1000) for x in vals])

target = np.log(train['SalePrice'])
fig = plt.figure()
ax = plt.axes()
(n, bins, patches) = plt.hist(target, 30, facecolor='dimgrey')
plt.xlabel('Log(Sale price)')


def doPlots(x, data, ii, fun):
    (fig, axes) = plt.subplots(len(ii) // 2, ncols=2)
    fig.tight_layout()
    for i in range(len(ii)):
        fun(x=x[ii[i]], data=data, ax=axes[i // 2, i % 2], color='dimgrey')

doPlots(cat_features, train, range(0, 8), sb.countplot)
doPlots(cat_features, train, range(8, 16), sb.countplot)
doPlots(cat_features, train, range(16, 24), sb.countplot)
doPlots(cat_features, train, range(24, 32), sb.countplot)
doPlots(cat_features, train, range(32, 40), sb.countplot)
doPlots(cat_features, train, range(40, 44), sb.countplot)
np.average(train['YearBuilt'][train['Foundation'] == 'PConc'])
np.average(train['YearBuilt'][train['Foundation'] == 'CBlock'])

def doHistPlots(x, data, ii, fun):
    (fig, axes) = plt.subplots(len(ii) // 2, ncols=2)
    fig.tight_layout()
    for i in range(len(ii)):
        fun(data[x[ii[i]]], color='b', ax=axes[i // 2, i % 2], kde=False)

doHistPlots(x=num_features, data=train, ii=range(0, 6), fun=sb.distplot)
doHistPlots(x=num_features, data=train, ii=range(6, 12), fun=sb.distplot)
doHistPlots(x=num_features, data=train, ii=range(12, 18), fun=sb.distplot)
doHistPlots(x=num_features, data=train, ii=range(18, 24), fun=sb.distplot)
train[train['TotalBsmtSF'] > 4000]
print(train['TotalBsmtSF'].iloc[1298,])
print(train['GrLivArea'].iloc[1298,])
print(train['OverallCond'].iloc[1298,])
print(train['OverallQual'].iloc[1298,])
trainX = trainX.drop(train.index[[1298]])
plt.rcParams['figure.figsize'] = (8.75, 7.0)
ax = plt.axes()
plot1 = sb.boxplot(data=train, x='Neighborhood', y='SalePrice')
ax.set_title('Price distribution by neighborhood')
sb.despine(offset=10, trim=True)
plt.xticks(rotation=90)


def transform_nb(x):
    if x in ('NoRidge', 'NridgHt', 'StoneBr'):
        return 5
    elif x in ('CollgCr', 'Veenker', 'Crawfor', 'Somerst', 'Timber', 'ClearCr'):
        return 4
    elif x in ('Mitchel', 'NWAmes', 'SawyerW', 'Gilbert', 'Blmngtn', 'SWISU', 'Blueste'):
        return 3
    elif x in ('OldTown', 'BrkSide', 'Sawyer', 'NAmes', 'IDOTRR', 'Edwards', 'BrDale', 'NPkVill'):
        return 2
    elif x in 'MeadowV':
        return 1
    else:
        return 9
train['NbdClass'] = train['Neighborhood'].apply(transform_nb)
test['NbdClass'] = test['Neighborhood'].apply(transform_nb)
trainX['NbdClass'] = train['NbdClass']
z = test.loc[:, 'NbdClass']
z = np.asarray(z)
testX.loc[:, 'NbdClass'] = z
trainX[trainX['NbdClass'] == 9]
plt.rcParams['figure.figsize'] = (8.75, 7.0)
ax = plt.axes()
plot1 = sb.boxplot(data=train, x='NbdClass', y='SalePrice')
ax.set_title('Price distribution by neighborhood')
sb.despine(offset=10, trim=True)
plt.xticks(rotation=90)

use_features = ['OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'GrLivArea', 'TotalBsmtSF', 'FullBath', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'NbdClass', 'HouseStyle', 'Alley', 'LotShape', 'LotConfig', 'Exterior1st', 'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtExposure', 'BsmtFinType1', 'HeatingQC', 'CentralAir', 'FireplaceQu', 'GarageType', 'GarageFinish']
alphas = np.logspace(-10, -1, num=20)
trainY = train['SalePrice']
trainY = trainY.drop(train.index[[1298]])
target = np.log(trainY)
lasso = linear_model.LassoCV(alphas=alphas, tol=0.0001, selection='random', random_state=17, max_iter=1000)