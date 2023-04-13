import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.ensemble import ExtraTreesRegressor
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
sns.set()
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
target = _input1['SalePrice']
all_df = pd.concat([_input1, _input0], ignore_index=True, sort=False)
all_df = all_df.drop('SalePrice', axis=1, inplace=False)
missings_df = {}
for key in all_df.columns:
    if all_df[key].isnull().sum() > 0:
        missings_df[key] = all_df[key].isnull().sum() / len(all_df[key]) * 100
missings_df = pd.DataFrame(missings_df, index=['MissingValues']).T.sort_values(by='MissingValues', ascending=False)
plt.figure(figsize=(15, 7), dpi=100)
plt.xticks(rotation=90)
sns.barplot(y=missings_df.MissingValues, x=missings_df.index, orient='v').set_title('The percentage of missing values per column')
all_df.describe(include='object').T.sort_values(by=['count']).head(10)
all_df[['FireplaceQu', 'Fireplaces']][all_df.FireplaceQu.isnull()]
sns.countplot(x=all_df['Fireplaces'])
plt.ylabel('The number of houses')
all_df[['GarageQual', 'GarageCars']][all_df.GarageQual.isnull()]
sns.countplot(x=all_df['GarageCars'])
plt.ylabel('The number of houses')
plt.xlabel('The car capacity of garage')
all_df['Alley'] = all_df['Alley'].fillna('NA', inplace=False)
all_df['PoolQC'] = all_df['PoolQC'].fillna('NA', inplace=False)
all_df['Fence'] = all_df['Fence'].fillna('NA', inplace=False)
all_df['MiscFeature'] = all_df['MiscFeature'].fillna('NA', inplace=False)
all_df['FireplaceQu'] = all_df['FireplaceQu'].fillna('NA', inplace=False)
all_df['GarageCond'] = all_df['GarageCond'].fillna('NA', inplace=False)
all_df['GarageQual'] = all_df['GarageQual'].fillna('NA', inplace=False)
all_df['GarageFinish'] = all_df['GarageFinish'].fillna('NA', inplace=False)
all_df['GarageType'] = all_df['GarageType'].fillna('NA', inplace=False)
all_df['BsmtExposure'] = all_df['BsmtExposure'].fillna('NA', inplace=False)
all_df['BsmtFinType2'] = all_df['BsmtFinType2'].fillna('NA', inplace=False)
all_df['BsmtFinType1'] = all_df['BsmtFinType1'].fillna('NA', inplace=False)
all_df['BsmtQual'] = all_df['BsmtQual'].fillna('NA', inplace=False)
all_df['BsmtCond'] = all_df['BsmtCond'].fillna('NA', inplace=False)
print('Number of missing values : ', all_df['MasVnrType'].isnull().sum())
print('Categories:', all_df['MasVnrType'].unique())
all_df['MasVnrType'] = all_df['MasVnrType'].fillna('None', inplace=False)
print('Number of missing values : ', all_df['Electrical'].isnull().sum())
print('Categories:', all_df['Electrical'].unique())
all_df['Electrical'] = all_df['Electrical'].fillna('SBrkr', inplace=False)
all_df['MSZoning'] = all_df['MSZoning'].fillna('RL', inplace=False)
all_df['Utilities'] = all_df['Utilities'].fillna('AllPub', inplace=False)
all_df['Functional'] = all_df['Functional'].fillna('Typ', inplace=False)
all_df['SaleType'] = all_df['SaleType'].fillna('WD', inplace=False)
all_df['Exterior2nd'] = all_df['Exterior2nd'].fillna('VinylSd', inplace=False)
all_df['Exterior1st'] = all_df['Exterior1st'].fillna('VinylSd', inplace=False)
all_df['KitchenQual'] = all_df['KitchenQual'].fillna('TA', inplace=False)
all_df.describe().T.sort_values(by='count').head(10)
(fig, ax) = plt.subplots(1, 2, figsize=(20, 5))
fig.suptitle('LotFrontage And LotArea Distribution', size=15)
sns.histplot(x=all_df.LotFrontage, kde=True, ax=ax[0])
sns.histplot(x=all_df.LotArea, kde=True, ax=ax[1])
Lot_tmp = all_df[['LotFrontage', 'LotArea']][~all_df.LotFrontage.isnull()]
z = np.abs(stats.zscore(Lot_tmp))
Lot_tmp_Z = Lot_tmp[(z < 3).all(axis=1)]
(fig, ax) = plt.subplots(1, 2, figsize=(20, 5))
fig.suptitle('LotFrontage And sqrt(LotArea) Distribution Without outliers', size=15)
sns.histplot(x=Lot_tmp_Z.LotFrontage, kde=True, ax=ax[0])
sns.histplot(x=Lot_tmp_Z.LotArea.apply(np.sqrt), kde=True, ax=ax[1])
plt.figure(figsize=(10, 5))
sns.regplot(x=Lot_tmp_Z.LotArea.apply(np.sqrt), y=Lot_tmp_Z.LotFrontage, line_kws={'color': 'red'}).set(title='Relation Between LotFrontage And Sqrt(LotArea)')
(fig, ax) = plt.subplots(1, 2, figsize=(20, 6))
sns.boxplot(x=all_df.BldgType, y=Lot_tmp_Z.LotFrontage, ax=ax[0]).set_title('BldgType (Type of dwelling) and LotFrontage', fontsize=20)
sns.boxplot(x=all_df.GarageCars, y=Lot_tmp_Z.LotFrontage, ax=ax[1]).set_title('GarageCars and LotFrontage', fontsize=20)
(fig, ax) = plt.subplots(2, 1, figsize=(25, 15))
sns.boxplot(x=all_df.Neighborhood, y=Lot_tmp_Z.LotFrontage, ax=ax[0]).set_title('LotFrontage And Neighborhood', fontsize=20)
sns.boxplot(x=all_df.Neighborhood, y=Lot_tmp_Z.LotArea.apply(np.sqrt), ax=ax[1]).set_title('LotArea And Neighborhood', fontsize=20)
(fig, ax) = plt.subplots(1, 1, figsize=(25, 8))
sns.boxplot(x=all_df[:1460].Neighborhood, y=target).set_title('Neighborhood And SalePrice', fontsize=20)
LotFrontage_df = pd.get_dummies(all_df[['LandSlope', 'BldgType', 'Alley', 'LotConfig', 'MSZoning', 'Neighborhood', 'LotArea', 'LotFrontage']].copy())
LotFrontage_df['LotArea'] = LotFrontage_df['LotArea'].apply(np.sqrt)
mask = ~LotFrontage_df.LotFrontage.isnull()
LotFrontage_train = LotFrontage_df.loc[mask]
LotFrontage_Test = LotFrontage_df.loc[~mask].drop('LotFrontage', axis=1)
z = np.abs(stats.zscore(LotFrontage_train[['LotFrontage', 'LotArea']]))
LotFrontage_train = LotFrontage_train[(z < 3).all(axis=1)]
Scaler_L = StandardScaler()
X = Scaler_L.fit_transform(LotFrontage_train.drop('LotFrontage', axis=1))
y = LotFrontage_train['LotFrontage']
(x_train, x_valid, y_train, y_valid) = train_test_split(X, y, test_size=0.3, random_state=42)
model = GradientBoostingRegressor(n_estimators=100, min_samples_split=26, min_samples_leaf=17, max_features='auto', max_depth=None)