import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost
import warnings
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from scipy.stats import norm, skew
from scipy.special import boxcox1p
pd.pandas.set_option('display.max_columns', None)
pd.pandas.set_option('display.max_rows', None)
warnings.filterwarnings('ignore')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
print('Shape of train data:', train.shape)
print('Shape of test data:', test.shape)
train_copy = train.copy()
test_copy = test.copy()
num_cols = train._get_numeric_data().columns

print()
print('Count: ', len(num_cols))
plt.figure(figsize=(25, 45))
for i in enumerate(num_cols):
    plt.subplot(13, 3, i[0] + 1)
    sns.boxplot(train[i[1]])
    plt.xlabel(i[1])
index = [712, 1219, 1416, 1200, 1345, 1458, 773, 1248, 1423, 628, 973, 1458, 1459]
train = train.drop(labels=index, axis=0)
print('Shape of train data:', train.shape)
print('Shape of test data:', test.shape)
Null_train = train.isnull().sum()
Null_train[Null_train > 0]
drop_columns = ['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'Id']
train = train.drop(drop_columns, axis=1)
test = test.drop(drop_columns, axis=1)
print('Shape of train data:', train.shape)
print('Shape of test data:', test.shape)
Null_train_data = train[['LotFrontage', 'FireplaceQu', 'MasVnrType', 'MasVnrArea', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond']]

def analysis(data):
    return pd.DataFrame({'Data Type': data.dtypes, 'Unique Count': data.apply(lambda x: x.nunique(), axis=0), 'Null Count': data.isnull().sum()})
analysis(Null_train_data)
Null_train_data[['LotFrontage', 'MasVnrArea', 'GarageYrBlt']].describe()
train['LotFrontage'] = train['LotFrontage'].fillna(train['LotFrontage'].mean())
train['MasVnrArea'] = train['MasVnrArea'].fillna(train['MasVnrArea'].mode()[0])
train['GarageYrBlt'] = train['GarageYrBlt'].fillna(train['GarageYrBlt'].mode()[0])
Null_test = test.isnull().sum()
Null_test[Null_test > 0]
Null_test_data = test[['MSZoning', 'LotFrontage', 'Utilities', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'KitchenQual', 'Functional', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'SaleType']]
analysis(Null_test_data)
Null_test_data[['LotFrontage', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt', 'GarageCars', 'GarageArea']].describe()
test['LotFrontage'] = test['LotFrontage'].fillna(test['LotFrontage'].mean())
test['MasVnrArea'] = test['MasVnrArea'].fillna(test['MasVnrArea'].mode()[0])
test['BsmtFinSF1'] = test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].mode()[0])
test['BsmtFinSF2'] = test['BsmtFinSF2'].fillna(test['BsmtFinSF2'].mode()[0])
test['BsmtUnfSF'] = test['BsmtUnfSF'].fillna(test['BsmtUnfSF'].mean())
test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].mean())
test['BsmtFullBath'] = test['BsmtFullBath'].fillna(test['BsmtFullBath'].mode()[0])
test['BsmtHalfBath'] = test['BsmtHalfBath'].fillna(test['BsmtHalfBath'].mode()[0])
test['GarageYrBlt'] = test['GarageYrBlt'].fillna(test['GarageYrBlt'].mode()[0])
test['GarageCars'] = test['GarageCars'].fillna(test['GarageCars'].mode()[0])
test['GarageArea'] = test['GarageArea'].fillna(test['GarageArea'].mean())

def correlation(data, limit):
    col = set()
    corr_matrix = data.corr()
    for i in range(len(corr_matrix)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > limit:
                col_name = corr_matrix.columns[i]
                col.add(col_name)
    return col
corr_columns = correlation(train, 0.7)
corr_columns
train = train.drop(['1stFlrSF', 'GarageArea', 'TotRmsAbvGrd'], axis=1)
test = test.drop(['1stFlrSF', 'GarageArea', 'TotRmsAbvGrd'], axis=1)
train.head()
House_Price = pd.DataFrame(train['SalePrice'])
train = train.drop(['SalePrice'], axis=1)
sns.displot(House_Price['SalePrice'], kde=True, color='Green')
sns.displot(np.log(House_Price['SalePrice']), kde=True, color='Black')
House_Price = pd.DataFrame(np.log(House_Price['SalePrice']))
print('Shape of train data:', train.shape)
print('Shape of test data:', test.shape)
data = pd.concat([train, test])
data.shape
data['YrBltRemod'] = data['YearBuilt'] + data['YearRemodAdd']
data['TotalBathrooms'] = data['FullBath'] + 0.5 * data['HalfBath'] + data['BsmtFullBath'] + 0.5 * data['BsmtHalfBath']
data['TotalPorchSf'] = data['OpenPorchSF'] + data['3SsnPorch'] + data['EnclosedPorch'] + data['ScreenPorch'] + data['WoodDeckSF']
data['TotalOutsideSF'] = sum((data['WoodDeckSF'], data['OpenPorchSF'], data['EnclosedPorch'], data['ScreenPorch']))
data['HouseAge'] = data['YrSold'] - data['YearBuilt']
data['OverallCondQual'] = (data['OverallCond'] + data['OverallQual']) / 2
data_num_cols = data._get_numeric_data().columns
data_num_cols
data_cat_cols = data.columns.difference(data_num_cols)
data_cat_cols
data_num_data = data.loc[:, data_num_cols]
data_cat_data = data.loc[:, data_cat_cols]
print('Shape of num data:', data_num_data.shape)
print('Shape of cat data:', data_cat_data.shape)
s_scaler = StandardScaler()
data_num_data_s = s_scaler.fit_transform(data_num_data)
data_num_data_s = pd.DataFrame(data_num_data_s, columns=data_num_cols)
data_cat_data = data_cat_data.fillna('NA')
label = LabelEncoder()
data_cat_data = data_cat_data.astype(str).apply(LabelEncoder().fit_transform)
data_num_data.isnull().sum()
data_num_data_s.reset_index(drop=True, inplace=True)
data_cat_data.reset_index(drop=True, inplace=True)
data_new = pd.concat([data_num_data_s, data_cat_data], axis=1)
train_new = data_new.loc[:1447,]
test_new = data_new.loc[1448:,]
print('Shape of train data:', train_new.shape)
print('Shape of test data:', test_new.shape)
from sklearn.model_selection import train_test_split
(trainx, valx, trainy, valy) = train_test_split(train_new, House_Price, test_size=0.2, random_state=1234)
print(trainx.shape)
print(valx.shape)
xgb = xgboost.XGBRegressor(base_score=0.5, booster='gbtree', colasample_bytree=0.2, colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1, gamma=0.1, importance_type='gain', learning_rate=0.1, max_delta_step=0, max_depth=10, min_child_weight=1, missing=1, n_estimators=100, n_jobs=1, nthread=None, objective='reg:linear', random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None, silent=None, subsample=1, verbosity=1)