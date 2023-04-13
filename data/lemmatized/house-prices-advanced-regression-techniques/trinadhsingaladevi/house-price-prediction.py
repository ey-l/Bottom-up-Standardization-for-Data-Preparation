import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
_input1.info()
_input1['SalePrice'].describe()
import seaborn as sns
import matplotlib.pyplot as plt
sns.distplot(_input1['SalePrice'])
plt.xticks(rotation=30)
print('Skewness = ', _input1['SalePrice'].skew())
corr = _input1.corr()
plt.figure(figsize=(15, 12))
sns.heatmap(corr, vmax=0.9, square=True)
plt.scatter(x=_input1['TotRmsAbvGrd'], y=_input1['GrLivArea'])
plt.xlabel('TotRmsAbvGrd')
plt.ylabel('GrLivArea')
plt.scatter(x=_input1['GarageYrBlt'], y=_input1['YearBuilt'])
plt.xlabel('GarageYrBlt')
plt.ylabel('YearBuilt')
plt.scatter(x=_input1['1stFlrSF'], y=_input1['TotalBsmtSF'])
plt.xlabel('1stFlrSF')
plt.ylabel('TotalBsmtSF')
plt.scatter(x=_input1['GarageCars'], y=_input1['GarageArea'])
plt.xlabel('GarageCars')
plt.ylabel('GarageArea')
corr = _input1.corr()
corr_top = corr['SalePrice'].sort_values(ascending=False)[:10]
top_features = corr_top.index[1:]
corr_top
numeric_cols = ['GrLivArea', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'YearBuilt']
nominal_cols = ['OverallQual', 'GarageCars', 'FullBath', 'TotRmsAbvGrd']
(fig, ax) = plt.subplots(nrows=9, ncols=1, figsize=(6, 30))
for i in range(len(top_features)):
    ax[i].scatter(x=_input1[top_features[i]], y=_input1['SalePrice'])
    ax[i].set_xlabel('%s' % top_features[i])
    ax[i].set_ylabel('SalePrice')
plt.tight_layout()
plt.savefig('./Top_featuresvsSalePrice.jpg', dpi=300, bbox_inches='tight')
Q1 = []
Q3 = []
Lower_bound = []
Upper_bound = []
Outliers = []
for i in top_features:
    (q1, q3) = (np.percentile(_input1[i], 25), np.percentile(_input1[i], 75))
    iqr = q3 - q1
    cut_off = 1.5 * iqr
    lower_bound = q1 - cut_off
    upper_bound = q3 + cut_off
    outlier = [x for x in _input1.index if _input1.loc[x, i] < lower_bound or _input1.loc[x, i] > upper_bound]
    Q1.append(q1)
    Q3.append(q3)
    Lower_bound.append(lower_bound)
    Upper_bound.append(upper_bound)
    Outliers.append(len(outlier))
    try:
        _input1 = _input1.drop(outlier, inplace=False, axis=0)
    except:
        continue
df_out = pd.DataFrame({'Column': top_features, 'Q1': Q1, 'Q3': Q3, 'Lower bound': Lower_bound, 'Upper_bound': Upper_bound, 'No. of outliers': Outliers})
df_out.sort_values(by='No. of outliers', ascending=False)
ntrain = _input1.shape[0]
target = np.log(_input1['SalePrice'])
_input1 = _input1.drop(['Id', 'SalePrice'], inplace=False, axis=1)
test_id = _input0['Id']
_input0 = _input0.drop('Id', inplace=False, axis=1)
_input1 = pd.concat([_input1, _input0])
_input1.isna().sum().sort_values(ascending=False).head(10)
_input1['PoolQC'].unique()
_input1['PoolQC'] = _input1['PoolQC'].replace(['Ex', 'Gd', 'TA', 'Fa', np.nan], [4, 3, 2, 1, 0], inplace=False)
_input1['Fence'] = _input1['Fence'].replace(['GdPrv', 'MnPrv', 'GdWo', 'MnWw', np.nan], [4, 3, 2, 1, 0], inplace=False)
_input1['FireplaceQu'] = _input1['FireplaceQu'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po', np.nan], [5, 4, 3, 2, 1, 0], inplace=False)
_input1['MiscFeature'] = _input1['MiscFeature'].fillna('None', inplace=False)
_input1['Alley'] = _input1['Alley'].fillna('None', inplace=False)
_input1['LotFrontage'] = _input1.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
_input1['GarageCars'].unique()
_input1['GarageYrBlt'].median()
for i in ['GarageCond', 'GarageQual']:
    _input1[i] = _input1[i].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po', np.nan], [5, 4, 3, 2, 1, 0], inplace=False)
_input1['GarageFinish'] = _input1['GarageFinish'].replace(['Fin', 'RFn', 'Unf', np.nan], [3, 2, 1, 0], inplace=False)
_input1['GarageType'] = _input1['GarageType'].fillna('None', inplace=False)
_input1['GarageYrBlt'] = _input1['GarageYrBlt'].fillna(_input1['GarageYrBlt'].median(), inplace=False)
_input1['GarageArea'] = _input1['GarageArea'].fillna(_input1['GarageYrBlt'].median(), inplace=False)
_input1['GarageCars'] = _input1['GarageCars'].fillna(0, inplace=False)
for i in ['BsmtCond', 'BsmtQual']:
    _input1[i] = _input1[i].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po', np.nan], [5, 4, 3, 2, 1, 0], inplace=False)
_input1['BsmtExposure'] = _input1['BsmtExposure'].replace(['Gd', 'Av', 'Mn', 'No', np.nan], [4, 3, 2, 1, 0], inplace=False)
for i in ['BsmtFinType1', 'BsmtFinType2']:
    _input1[i] = _input1[i].replace(['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', np.nan], [6, 5, 4, 3, 2, 1, 0], inplace=False)
for i in ['BsmtHalfBath', 'BsmtFullBath', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']:
    _input1[i] = _input1[i].fillna(0, inplace=False)
_input1['MasVnrType'] = _input1['MasVnrType'].fillna('None', inplace=False)
_input1['MasVnrArea'] = _input1['MasVnrArea'].fillna(0, inplace=False)
for i in ['MSZoning', 'Utilities']:
    _input1[i] = _input1[i].fillna(_input1[i].mode()[0], inplace=False)
_input1['Functional'] = _input1['Functional'].fillna('Typ', inplace=False)
_input1['SaleType'] = _input1['SaleType'].fillna('Oth', inplace=False)
_input1['Electrical'] = _input1['Electrical'].fillna(_input1['Electrical'].mode()[0], inplace=False)
for i in ['Exterior1st', 'Exterior2nd']:
    _input1[i] = _input1[i].fillna('Other', inplace=False)
_input1['KitchenQual'] = _input1['KitchenQual'].fillna(_input1['KitchenQual'].mode()[0], inplace=False)
_input1['KitchenQual'] = _input1['KitchenQual'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po'], [4, 3, 2, 1, 0], inplace=False)
_input1['CentralAir'] = _input1['CentralAir'].replace(['N', 'Y'], [0, 1], inplace=False)
for i in ['HeatingQC', 'ExterCond', 'ExterQual']:
    _input1[i] = _input1[i].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po'], [4, 3, 2, 1, 0], inplace=False)
_input1['TotalSF'] = _input1.apply(lambda x: x['1stFlrSF'] + x['2ndFlrSF'] + x['TotalBsmtSF'], axis=1)
_input1['TotalBath'] = _input1.apply(lambda x: x['FullBath'] + 0.5 * x['HalfBath'] + x['BsmtFullBath'] + 0.5 * x['BsmtHalfBath'], axis=1)
_input1['TotalPorch'] = _input1.apply(lambda x: x['OpenPorchSF'] + x['EnclosedPorch'] + x['3SsnPorch'] + x['ScreenPorch'], axis=1)
_input1['NewHouse'] = _input1.apply(lambda x: 1 if x['SaleCondition'] == 'Partial' else 0, axis=1)
_input1 = pd.get_dummies(_input1, drop_first=True)
_input1.head()
df = _input1.iloc[:ntrain, :]
_input0 = _input1.iloc[ntrain:, :]
from sklearn.model_selection import train_test_split
X = df
y = target
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=27)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
lr = LinearRegression()