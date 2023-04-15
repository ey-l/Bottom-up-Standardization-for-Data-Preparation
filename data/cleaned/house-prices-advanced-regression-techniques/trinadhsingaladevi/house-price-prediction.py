import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train.head()
train.info()
train['SalePrice'].describe()
import seaborn as sns
import matplotlib.pyplot as plt
sns.distplot(train['SalePrice'])
plt.xticks(rotation=30)
print('Skewness = ', train['SalePrice'].skew())
corr = train.corr()
plt.figure(figsize=(15, 12))
sns.heatmap(corr, vmax=0.9, square=True)

plt.scatter(x=train['TotRmsAbvGrd'], y=train['GrLivArea'])
plt.xlabel('TotRmsAbvGrd')
plt.ylabel('GrLivArea')

plt.scatter(x=train['GarageYrBlt'], y=train['YearBuilt'])
plt.xlabel('GarageYrBlt')
plt.ylabel('YearBuilt')

plt.scatter(x=train['1stFlrSF'], y=train['TotalBsmtSF'])
plt.xlabel('1stFlrSF')
plt.ylabel('TotalBsmtSF')

plt.scatter(x=train['GarageCars'], y=train['GarageArea'])
plt.xlabel('GarageCars')
plt.ylabel('GarageArea')

corr = train.corr()
corr_top = corr['SalePrice'].sort_values(ascending=False)[:10]
top_features = corr_top.index[1:]
corr_top
numeric_cols = ['GrLivArea', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'YearBuilt']
nominal_cols = ['OverallQual', 'GarageCars', 'FullBath', 'TotRmsAbvGrd']
(fig, ax) = plt.subplots(nrows=9, ncols=1, figsize=(6, 30))
for i in range(len(top_features)):
    ax[i].scatter(x=train[top_features[i]], y=train['SalePrice'])
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
    (q1, q3) = (np.percentile(train[i], 25), np.percentile(train[i], 75))
    iqr = q3 - q1
    cut_off = 1.5 * iqr
    lower_bound = q1 - cut_off
    upper_bound = q3 + cut_off
    outlier = [x for x in train.index if train.loc[x, i] < lower_bound or train.loc[x, i] > upper_bound]
    Q1.append(q1)
    Q3.append(q3)
    Lower_bound.append(lower_bound)
    Upper_bound.append(upper_bound)
    Outliers.append(len(outlier))
    try:
        train.drop(outlier, inplace=True, axis=0)
    except:
        continue
df_out = pd.DataFrame({'Column': top_features, 'Q1': Q1, 'Q3': Q3, 'Lower bound': Lower_bound, 'Upper_bound': Upper_bound, 'No. of outliers': Outliers})
df_out.sort_values(by='No. of outliers', ascending=False)
ntrain = train.shape[0]
target = np.log(train['SalePrice'])
train.drop(['Id', 'SalePrice'], inplace=True, axis=1)
test_id = test['Id']
test.drop('Id', inplace=True, axis=1)
train = pd.concat([train, test])
train.isna().sum().sort_values(ascending=False).head(10)
train['PoolQC'].unique()
train['PoolQC'].replace(['Ex', 'Gd', 'TA', 'Fa', np.nan], [4, 3, 2, 1, 0], inplace=True)
train['Fence'].replace(['GdPrv', 'MnPrv', 'GdWo', 'MnWw', np.nan], [4, 3, 2, 1, 0], inplace=True)
train['FireplaceQu'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po', np.nan], [5, 4, 3, 2, 1, 0], inplace=True)
train['MiscFeature'].fillna('None', inplace=True)
train['Alley'].fillna('None', inplace=True)
train['LotFrontage'] = train.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
train['GarageCars'].unique()
train['GarageYrBlt'].median()
for i in ['GarageCond', 'GarageQual']:
    train[i].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po', np.nan], [5, 4, 3, 2, 1, 0], inplace=True)
train['GarageFinish'].replace(['Fin', 'RFn', 'Unf', np.nan], [3, 2, 1, 0], inplace=True)
train['GarageType'].fillna('None', inplace=True)
train['GarageYrBlt'].fillna(train['GarageYrBlt'].median(), inplace=True)
train['GarageArea'].fillna(train['GarageYrBlt'].median(), inplace=True)
train['GarageCars'].fillna(0, inplace=True)
for i in ['BsmtCond', 'BsmtQual']:
    train[i].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po', np.nan], [5, 4, 3, 2, 1, 0], inplace=True)
train['BsmtExposure'].replace(['Gd', 'Av', 'Mn', 'No', np.nan], [4, 3, 2, 1, 0], inplace=True)
for i in ['BsmtFinType1', 'BsmtFinType2']:
    train[i].replace(['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', np.nan], [6, 5, 4, 3, 2, 1, 0], inplace=True)
for i in ['BsmtHalfBath', 'BsmtFullBath', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']:
    train[i].fillna(0, inplace=True)
train['MasVnrType'].fillna('None', inplace=True)
train['MasVnrArea'].fillna(0, inplace=True)
for i in ['MSZoning', 'Utilities']:
    train[i].fillna(train[i].mode()[0], inplace=True)
train['Functional'].fillna('Typ', inplace=True)
train['SaleType'].fillna('Oth', inplace=True)
train['Electrical'].fillna(train['Electrical'].mode()[0], inplace=True)
for i in ['Exterior1st', 'Exterior2nd']:
    train[i].fillna('Other', inplace=True)
train['KitchenQual'].fillna(train['KitchenQual'].mode()[0], inplace=True)
train['KitchenQual'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po'], [4, 3, 2, 1, 0], inplace=True)
train['CentralAir'].replace(['N', 'Y'], [0, 1], inplace=True)
for i in ['HeatingQC', 'ExterCond', 'ExterQual']:
    train[i].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po'], [4, 3, 2, 1, 0], inplace=True)
train['TotalSF'] = train.apply(lambda x: x['1stFlrSF'] + x['2ndFlrSF'] + x['TotalBsmtSF'], axis=1)
train['TotalBath'] = train.apply(lambda x: x['FullBath'] + 0.5 * x['HalfBath'] + x['BsmtFullBath'] + 0.5 * x['BsmtHalfBath'], axis=1)
train['TotalPorch'] = train.apply(lambda x: x['OpenPorchSF'] + x['EnclosedPorch'] + x['3SsnPorch'] + x['ScreenPorch'], axis=1)
train['NewHouse'] = train.apply(lambda x: 1 if x['SaleCondition'] == 'Partial' else 0, axis=1)
train = pd.get_dummies(train, drop_first=True)
train.head()
df = train.iloc[:ntrain, :]
test = train.iloc[ntrain:, :]
from sklearn.model_selection import train_test_split
X = df
y = target
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=27)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
lr = LinearRegression()