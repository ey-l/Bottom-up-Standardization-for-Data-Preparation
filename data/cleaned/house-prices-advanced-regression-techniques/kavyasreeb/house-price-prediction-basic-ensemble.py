import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import norm, skew
import pylab
import warnings

def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train.head(3)
train.shape
UniqueIds = len(set(train.Id))
idsTotal = train.shape[0]
idsDupli = idsTotal - UniqueIds
print('There are ' + str(idsDupli) + ' duplicate IDs for ' + str(idsTotal) + ' total entries')
train_ID = train['Id']
test_ID = test['Id']
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)
train.info()
numeric_var = train.select_dtypes(exclude=['object']).drop(['MSSubClass', 'SalePrice'], axis=1).copy()
numeric_var.columns
disc_num_var = ['OverallQual', 'OverallCond', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'MoSold', 'YrSold']
cont_num_var = []
for i in numeric_var.columns:
    if i not in disc_num_var:
        cont_num_var.append(i)
print('Continuous features are:\n', cont_num_var, '\n')
print('Discrete features are:\n', disc_num_var)
cat_train = train.select_dtypes(include=['object']).copy()
cat_train['MSSubClass'] = train['MSSubClass']
cat_train.columns
fig = plt.figure(figsize=(14, 15))
for (index, col) in enumerate(cont_num_var):
    plt.subplot(8, 3, index + 1)
    sns.histplot(numeric_var.loc[:, col].dropna(), kde=False)
fig.tight_layout(pad=1.0)
fig = plt.figure(figsize=(14, 15))
for (index, col) in enumerate(cont_num_var):
    plt.subplot(6, 4, index + 1)
    sns.boxplot(y=col, data=numeric_var.dropna())
fig.tight_layout(pad=1.0)
fig = plt.figure(figsize=(14, 15))
for (index, col) in enumerate(disc_num_var):
    plt.subplot(5, 3, index + 1)
    sns.countplot(x=col, data=numeric_var.dropna())
fig.tight_layout(pad=1.0)
fig = plt.figure(figsize=(20, 20))
for index in range(len(cat_train.columns)):
    plt.subplot(9, 5, index + 1)
    sns.countplot(x=cat_train.iloc[:, index], data=cat_train.dropna())
    plt.xticks(rotation=90)
fig.tight_layout(pad=1.0)
corrmat = train.corr()
(f, ax) = plt.subplots(figsize=(14, 10))
sns.heatmap(corrmat, mask=corrmat < 0.5, vmax=0.8, square=True, cmap='Reds')
k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values, cmap='Reds')

train['SalePrice'].describe()
sns.displot(train['SalePrice'], kde=True)
print('Skewness: %f' % train['SalePrice'].skew())
print('Kurtosis: %f' % train['SalePrice'].kurt())
train['SalePrice'] = np.log1p(train['SalePrice'])
sns.displot(train['SalePrice'], kde=True)
stats.probplot(train.SalePrice, plot=pylab)
out_col = ['LotFrontage', 'LotArea', 'BsmtFinSF1', 'TotalBsmtSF', 'GrLivArea']
fig = plt.figure(figsize=(20, 5))
for (index, col) in enumerate(out_col):
    plt.subplot(1, 5, index + 1)
    sns.boxplot(y=col, data=train)
fig.tight_layout(pad=1.5)
train = train.drop(train[train['TotalBsmtSF'] > 5000].index)
train = train.drop(train[train['GrLivArea'] > 4000].index)
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print('all_data size is : {}'.format(all_data.shape))
all_data.drop(['GarageYrBlt', 'TotRmsAbvGrd', 'GarageArea'], axis=1, inplace=True)
total = all_data.isnull().sum().sort_values(ascending=False)
percent = (100 * all_data.isnull().sum() / all_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data = missing_data.head(35)
missing_data.style.background_gradient(cmap='Reds')
all_data['PoolQC'] = all_data['PoolQC'].fillna('None')
all_data['MiscFeature'] = all_data['MiscFeature'].fillna('None')
all_data['Alley'] = all_data['Alley'].fillna('None')
all_data['Fence'] = all_data['Fence'].fillna('None')
all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna('None')
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
all_data['GarageCars'] = all_data['GarageCars'].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
all_data['MasVnrType'] = all_data['MasVnrType'].fillna('None')
all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data['Functional'] = all_data['Functional'].fillna('Typ')
all_data = all_data.drop(['Utilities'], axis=1)
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
all_data_na = all_data.isnull().sum() / len(all_data) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio': all_data_na})
missing_data.head()
all_data = all_data.drop(['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF'], axis=1)
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['TotalBath'] = all_data['BsmtFullBath'] + all_data['FullBath'] + all_data['HalfBath']
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
all_data['OverallCond'] = all_data['OverallCond'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)
numeric_features = all_data.dtypes[all_data.dtypes != 'object'].index
skewed_feats = all_data[numeric_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print('Skew in numerical features: \n')
skewness = pd.DataFrame({'Skew': skewed_feats})
skewness.head()
skewness = skewness[abs(skewness) > 0.5]
print('There are {} skewed numerical features to Box Cox transform'.format(skewness.shape[0]))
from scipy.special import boxcox1p
skewed_features = skewness.index
lmbd = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lmbd)
from sklearn.preprocessing import LabelEncoder
cols = ('LotShape', 'LandSlope', 'Street', 'Alley', 'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence')
for c in cols:
    lbl = LabelEncoder()