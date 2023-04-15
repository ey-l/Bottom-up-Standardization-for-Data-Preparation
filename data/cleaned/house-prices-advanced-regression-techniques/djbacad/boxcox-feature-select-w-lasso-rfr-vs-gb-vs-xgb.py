import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from scipy.stats import norm, skew
from scipy import stats
from scipy.special import boxcox1p
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
HousingPrices_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', sep=',', header=0)
HousingPrices_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', sep=',', header=0)
HousingPrices_test['SalePrice'] = 0
print('Train Shape: ' + str(HousingPrices_train.shape))
print('Test Shape: ' + str(HousingPrices_test.shape))
HousingPrices_full = pd.concat([HousingPrices_train, HousingPrices_test])
HousingPrices_full.shape
HousingPrices_full.head(5)
HousingPrices_full.tail(5)
HousingPrices_full.describe()
HousingPrices_full.replace('', np.nan, inplace=True)
HousingPrices_full['MSSubClass'] = HousingPrices_full['MSSubClass'].astype('object')
HousingPrices_full['OverallCond'] = HousingPrices_full['OverallCond'].astype('object')
HousingPrices_full['YrSold'] = HousingPrices_full['YrSold'].astype('object')
HousingPrices_full['MoSold'] = HousingPrices_full['MoSold'].astype('object')
sns.set_palette('GnBu_d')
plt.title('Missingess Map')
plt.rcParams['figure.figsize'] = (40, 40)
sns.heatmap(HousingPrices_full.isnull(), cbar=False)
for (i, j) in HousingPrices_full.isnull().sum().iteritems():
    print(i, j)
HousingPrices_full['LotFrontage'].fillna(value=HousingPrices_full['LotFrontage'].median(), inplace=True)
HousingPrices_full.drop('FireplaceQu', axis=1, inplace=True)
d = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
for i in d:
    HousingPrices_full[i] = HousingPrices_full[i].fillna('None')
d = ['GarageArea', 'GarageYrBlt', 'GarageCars']
for i in d:
    HousingPrices_full[i] = HousingPrices_full[i].fillna(0)
d = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']
for i in d:
    HousingPrices_full[i] = HousingPrices_full[i].fillna(0)
d = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
for i in d:
    HousingPrices_full[i] = HousingPrices_full[i].fillna('None')
HousingPrices_full['PoolQC'].fillna('None', inplace=True)
HousingPrices_full['MiscFeature'].fillna('None', inplace=True)
HousingPrices_full['Alley'].fillna('None', inplace=True)
HousingPrices_full['Fence'].fillna('None', inplace=True)
HousingPrices_full['MasVnrArea'] = HousingPrices_full['MasVnrArea'].fillna(0)
HousingPrices_full['MasVnrType'] = HousingPrices_full['MasVnrType'].fillna('None')
HousingPrices_full['MSZoning'].fillna(value=HousingPrices_full['MSZoning'].value_counts().idxmax(), inplace=True)
HousingPrices_full.drop('Utilities', axis=1, inplace=True)
HousingPrices_full['Functional'].fillna('Typ', inplace=True)
HousingPrices_full['KitchenQual'].fillna(value=HousingPrices_full['KitchenQual'].value_counts().idxmax(), inplace=True)
HousingPrices_full['Electrical'].fillna(value=HousingPrices_full['Electrical'].value_counts().idxmax(), inplace=True)
HousingPrices_full['Exterior1st'].fillna(value=HousingPrices_full['Exterior1st'].value_counts().idxmax(), inplace=True)
HousingPrices_full['Exterior2nd'].fillna(value=HousingPrices_full['Exterior2nd'].value_counts().idxmax(), inplace=True)
HousingPrices_full['SaleType'].fillna(value=HousingPrices_full['SaleType'].value_counts().idxmax(), inplace=True)
HousingPrices_full.info()
plt.rcParams['figure.figsize'] = (20, 20)
sns.heatmap(HousingPrices_full.isnull(), cbar=False)
HousingPrices_train = HousingPrices_full[0:1460]
HousingPrices_test = HousingPrices_full[1460:2919]
full_numeric = HousingPrices_full.dtypes[HousingPrices_full.dtypes != 'object'].index
skewed_features = HousingPrices_full[full_numeric].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print('\nSkewness of Numeric Features \n')
skewness = pd.DataFrame({'skew': skewed_features})
skewness
plt.rcParams['figure.figsize'] = (7.0, 5.0)
sns.distplot(HousingPrices_full['LotFrontage'], fit=norm)
fig = plt.figure()
res = stats.probplot(HousingPrices_full['LotFrontage'], plot=plt)
plt.rcParams['figure.figsize'] = (7.0, 5.0)
sns.distplot(HousingPrices_full['LotArea'], fit=norm)
fig = plt.figure()
res = stats.probplot(HousingPrices_full['LotArea'], plot=plt)
to_transform = skewness[abs(skewness) > 0.7]
to_transform = to_transform[to_transform['skew'].notna()]
to_transform
greater070 = list(to_transform.index)
greater070.remove('SalePrice')
for i in greater070:
    HousingPrices_full[i] = boxcox1p(HousingPrices_full[i], 0.15)
plt.rcParams['figure.figsize'] = (7.0, 5.0)
sns.distplot(HousingPrices_full['LotFrontage'], fit=norm)
fig = plt.figure()
res = stats.probplot(HousingPrices_full['LotFrontage'], plot=plt)
plt.rcParams['figure.figsize'] = (7.0, 5.0)
sns.distplot(HousingPrices_full['LotArea'], fit=norm)
fig = plt.figure()
res = stats.probplot(HousingPrices_full['LotArea'], plot=plt)
plt.rcParams['figure.figsize'] = (7.0, 5.0)
sns.distplot(HousingPrices_train['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(HousingPrices_train['SalePrice'], plot=plt)
HousingPrices_train['SalePrice'] = np.log(HousingPrices_train['SalePrice'])
sns.distplot(HousingPrices_train['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(HousingPrices_train['SalePrice'], plot=plt)
plt.rcParams['figure.figsize'] = (15.0, 13.0)
plt.title('Correlation Plot')
sns.heatmap(HousingPrices_train.corr())
HousingPrices_full['TotalSF'] = HousingPrices_full['BsmtFinSF1'] + HousingPrices_full['BsmtFinSF2'] + HousingPrices_full['1stFlrSF'] + HousingPrices_full['2ndFlrSF']
todummify = list(HousingPrices_full.select_dtypes(include=['object']).columns)
HousingPrices_full = pd.get_dummies(HousingPrices_full, columns=todummify)
tocategorify = list(HousingPrices_full.select_dtypes(include=['uint8']).columns)
HousingPrices_full[tocategorify] = HousingPrices_full[tocategorify].astype('category')
HousingPrices_full.info(verbose=True)
HousingPrices_train = HousingPrices_full[0:1460]
HousingPrices_test = HousingPrices_full[1460:2919]
X = HousingPrices_train.drop(['SalePrice', 'Id'], axis=1)
y = HousingPrices_train['SalePrice']
print('Dependent Variables')

print('Independent Variable')

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_numeric = X._get_numeric_data()
X_copy = X.copy()
X_copy_numeric_scaled = pd.DataFrame(scaler.fit_transform(X_numeric), columns=X_numeric.columns)
X_copy.update(X_copy_numeric_scaled)
X_copy[tocategorify] = X_copy[tocategorify].astype('float')
from sklearn.linear_model import LassoCV
reg = LassoCV(cv=10)