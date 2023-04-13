import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
testID = _input0['Id']
data = pd.concat([_input1.drop('SalePrice', axis=1), _input0], keys=['train', 'test'])
data = data.drop(['Id'], axis=1, inplace=False)
data.head(2)
years = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']
metrics = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
data[years].max()
mask = (data[years] > 2018).any(axis=1)
data[mask]
data.loc[mask, 'GarageYrBlt'] = data[mask]['YearBuilt']
mask = (data[metrics] < 0).any(axis=1)
data[mask]
mask = (data['MoSold'] > 12) | (data['MoSold'] < 1)
data[mask]
num_feats = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'HeatingQC', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Fireplaces', 'FireplaceQu', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'MiscVal', 'YrSold']
grades = ['OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
literal = ['Ex', 'Gd', 'TA', 'Fa', 'Po']
num = [9, 7, 5, 3, 2]
G = dict(zip(literal, num))
data[grades] = data[grades].replace(G)
cat_feats = data.drop(num_feats, axis=1).columns
cat_feats
price = np.log1p(_input1['SalePrice'])
skewed_feats = data.loc['train'][metrics].apply(lambda x: x.skew(skipna=True))
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
data[skewed_feats] = np.log1p(data[skewed_feats])
data.isnull().sum()[data.isnull().sum() > 0]
feats = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'Electrical', 'Functional', 'SaleType']
model = data.loc['train'].groupby('Neighborhood')[feats].apply(lambda x: x.mode().iloc[0])
for f in feats:
    data[f] = data[f].fillna(data['Neighborhood'].map(model[f]), inplace=False)
plt.subplots(figsize=(15, 5))
boxdata = data.loc['train'].groupby('LotConfig')['LotFrontage'].median().sort_values(ascending=False)
order = boxdata.index
sns.boxplot(x='LotConfig', y='LotFrontage', order=order, data=data.loc['train'])
data['LotFrontage'] = data['LotFrontage'].fillna(data.loc['train', 'LotFrontage'].median())
data['KitchenQual'] = data['KitchenQual'].fillna(data['OverallQual'], inplace=False)
bsmt = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'BsmtFullBath', 'BsmtHalfBath', 'TotalBsmtSF']
fire = ['Fireplaces', 'FireplaceQu']
garage = ['GarageQual', 'GarageCond', 'GarageType', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageYrBlt']
masn = ['MasVnrType', 'MasVnrArea']
others = ['Alley', 'Fence', 'PoolQC', 'MiscFeature']
cats = data.columns[data.dtypes == 'object']
nums = list(set(data.columns) - set(cats))
data['MasVnrType'] = data['MasVnrType'].replace({'None': np.nan}, inplace=False)
data[cats] = data[cats].fillna('0')
data[nums] = data[nums].fillna(0)
data.isnull().sum().sum()
data['MSSubClass'] = data['MSSubClass'].astype('object', copy=False)
data['MoSold'] = data['MoSold'].astype('object', copy=False)
data['BsmtFullBath'] = data['BsmtFullBath'].astype('int64', copy=False)
data['BsmtHalfBath'] = data['BsmtHalfBath'].astype('int64', copy=False)
data['GarageCars'] = data['GarageCars'].astype('int64', copy=False)
data[years] = data[years].astype('int64', copy=False)
categorical_data = pd.concat((data.loc['train'][cat_feats], price), axis=1)
low = 0.05 * data.loc['train'].shape[0]
for feat in cat_feats:
    order = categorical_data.groupby(feat).mean().sort_values(by='SalePrice', ascending=False).index.values.tolist()
    for i in range(0, len(order)):
        N = categorical_data[categorical_data[feat] == order[i]].count().max()
        j = i
        while (N < low) & (N != 0):
            j += 1
            if j > len(order) - 1:
                j = i - 1
                break
            else:
                N += categorical_data[categorical_data[feat] == order[j]].count().max()
        if j < i:
            lim = len(order)
        else:
            lim = j
        for k in range(i, lim):
            categorical_data = categorical_data.replace({feat: {order[k]: order[j]}}, inplace=False)
            data = data.replace({feat: {order[k]: order[j]}}, inplace=False)
    uniD = data[feat].unique()
    order = categorical_data[feat].unique()
    for i in uniD:
        if i not in order:
            ind = np.argsort(order - i)[0]
            data = data.replace({feat: {i: order[ind]}}, inplace=False)
data.columns
for feat in categorical_data.columns[:-1]:
    uni = categorical_data.groupby(feat).mean().sort_values(by='SalePrice').index
    if len(uni) < 2:
        data = data.drop(feat, axis=1, inplace=False)
    elif len(uni) < 3:
        print('{}: {}'.format(feat, uni))
        data[feat] = data[feat].replace({uni[0]: 0, uni[1]: 1}, inplace=False)
        data[feat] = data[feat].astype('int8')
    else:
        data[feat] = data[feat].astype('category')
finaldata = pd.get_dummies(data)
black_list = bsmt + fire + garage + masn + others
for feat in finaldata.columns:
    if '_0' in feat and feat.split('_')[0] in black_list:
        finaldata = finaldata.drop(feat, axis=1, inplace=False)
finaldata.shape
X_test = finaldata.loc['test']
X_train = finaldata.loc['train']
y_train = price
m = X_train.mean()
std = X_train.std()
X_train = (X_train - m) / std
X_test = (X_test - m) / std
LR = LinearRegression()