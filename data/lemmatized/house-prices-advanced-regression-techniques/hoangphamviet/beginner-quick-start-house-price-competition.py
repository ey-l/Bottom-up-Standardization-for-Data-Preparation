import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
import missingno as msno
from scipy import stats
from sklearn import preprocessing
from sklearn import feature_selection
import warnings
warnings.filterwarnings('ignore')
SEED = 42

def concat_df(train_data, test_data):
    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)

def divide_df(all_data):
    return (all_data.loc[:1459], all_data.loc[1460:])
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
y_train = _input1.SalePrice
id_val = _input1.Id
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
df_all = concat_df(_input1, _input0).drop(['SalePrice', 'Id'], axis=1)
_input1.name = 'Training Set'
_input0.name = 'Test Set'
df_all.name = 'All Set'
dfs = [_input1, _input0]
df_all.head()
for df in dfs:
    print(f'Only features contained missing value in {df.name}')
    temp = df.isnull().sum()
    print(temp.loc[temp != 0], '\n')
null_features = df_all.isnull().sum()
null_100 = df_all.columns[list((null_features < 100) & (null_features != 0))]
num = df_all[null_100].select_dtypes(include=np.number).columns
non_num = df_all[null_100].select_dtypes(include='object').columns
df_all[num] = df_all[num].apply(lambda x: x.fillna(x.median()))
df_all[non_num] = df_all[non_num].apply(lambda x: x.fillna(x.value_counts().index[0]))
null_1000 = df_all.columns[list(null_features > 1000)]
df_all = df_all.drop(null_1000, axis=1, inplace=False)
df_all = df_all.drop(['GarageYrBlt', 'LotFrontage'], axis=1, inplace=False)
df_all['GarageCond'] = df_all['GarageCond'].fillna('Null')
df_all['GarageFinish'] = df_all['GarageFinish'].fillna('Null')
df_all['GarageQual'] = df_all['GarageQual'].fillna('Null')
df_all['GarageType'] = df_all['GarageType'].fillna('Null')
(_input1, _input0) = divide_df(df_all)
_input1 = pd.concat([_input1, y_train], axis=1)
print(df_all.isnull().any().sum())
df_all['YearBuilt'] = pd.qcut(df_all['YearBuilt'], 10, duplicates='drop')
df_all['YearRemodAdd'] = pd.qcut(df_all['YearRemodAdd'], 10, duplicates='drop')
df_all['YrSold'] = pd.qcut(df_all['YrSold'], 10, duplicates='drop')
for cate_col in ['YearBuilt', 'YearRemodAdd', 'YrSold']:
    df_all[cate_col] = preprocessing.LabelEncoder().fit_transform(df_all[cate_col].values)
(_input1, _input0) = divide_df(df_all)
df_all['TotalPorchSF'] = df_all['OpenPorchSF'] + df_all['3SsnPorch'] + df_all['EnclosedPorch'] + df_all['ScreenPorch'] + df_all['WoodDeckSF']
df_all['HasGarage'] = df_all['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
df_all['TotalBath'] = df_all['FullBath'] + 0.5 * df_all['HalfBath'] + df_all['BsmtFullBath'] + 0.5 * df_all['BsmtHalfBath']
df_all['HasFireplace'] = df_all['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
df_all['TotalBsmtbath'] = df_all['BsmtFullBath'] + 0.5 * df_all['BsmtHalfBath']
df_all['TotalSF'] = df_all['BsmtFinSF1'] + df_all['BsmtFinSF2'] + df_all['1stFlrSF'] + df_all['2ndFlrSF']
df_all = df_all.drop(['OpenPorchSF', '3SsnPorch', 'EnclosedPorch', 'ScreenPorch', 'WoodDeckSF', 'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath'], axis=1, inplace=False)
num_features = ['OverallQual', 'GrLivArea', 'TotalSF', 'GarageCars', 'TotalBath', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'TotRmsAbvGrd', 'MasVnrArea', 'HasFireplace', 'Fireplaces', 'TotalPorchSF', '2ndFlrSF', 'LotArea', 'HasGarage', 'TotalBsmtbath', 'BsmtUnfSF', 'YearBuilt', 'YearRemodAdd', 'YrSold']
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
num_cols = df_all.select_dtypes(include=numeric_dtypes).columns
drop_num = np.setdiff1d(num_cols, num_features)
df_all = df_all.drop(drop_num, axis=1, inplace=False)
skew_features = df_all[num_features].apply(lambda x: stats.skew(x)).sort_values(ascending=False)
skew_features = skew_features[abs(skew_features) > 0.5]
print(skew_features)
for feat in skew_features.index:
    df_all[feat] = np.log1p(df_all[feat])
(_input1, _input0) = divide_df(df_all)
_input1[num_features].head()
_input1['Electrical'].loc[_input1['Electrical'] == 'Mix'] = 'SBrkr'
_input1['Exterior2nd'].loc[_input1['Exterior2nd'] == 'Other'] = 'VinylSd'
_input1['Heating'].loc[_input1['Heating'] == 'OthW'] = 'GasA'
_input1['Heating'].loc[_input1['Heating'] == 'Floor'] = 'GasA'
_input1['HouseStyle'].loc[_input1['HouseStyle'] == '2.5Fin'] = '1.5Fin'
cate_features = ['BldgType', 'BsmtExposure', 'BsmtFinType1', 'BsmtQual', 'CentralAir', 'Condition1', 'Electrical', 'ExterCond', 'ExterQual', 'Exterior2nd', 'Functional', 'GarageCond', 'GarageType', 'Heating', 'HouseStyle', 'KitchenQual', 'LandContour', 'LandSlope', 'LotShape', 'Neighborhood', 'PavedDrive', 'RoofStyle', 'SaleCondition', 'SaleType', 'Street', 'YearBuilt', 'YearRemodAdd', 'YrSold']
cols = _input1.select_dtypes(include=['object', 'category']).columns
drop_cate = np.setdiff1d(cols, cate_features)
_input1 = _input1.drop(drop_cate, axis=1, inplace=False)
_input0 = _input0.drop(drop_cate, axis=1, inplace=False)
print(_input1.shape, _input0.shape)
encoded_features = list()
for df in [_input1, _input0]:
    for feature in cate_features:
        encoded_feat = preprocessing.OneHotEncoder().fit_transform(df[feature].values.reshape(-1, 1)).toarray()
        n = df[feature].nunique()
        cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]
        encoded_df = pd.DataFrame(encoded_feat, columns=cols)
        encoded_df.index = df.index
        encoded_features.append(encoded_df)
_input1 = pd.concat([_input1, *encoded_features[:len(cate_features)]], axis=1)
_input0 = pd.concat([_input0, *encoded_features[len(cate_features):]], axis=1)
print(_input1.shape, _input0.shape)
_input1 = _input1.drop(cate_features, axis=1, inplace=False)
_input0 = _input0.drop(cate_features, axis=1, inplace=False)
df_all = concat_df(_input1, _input0)
print(_input1.shape, _input0.shape)
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
kfolds = KFold(n_splits=10, shuffle=True, random_state=SEED)

def evaluate_model_cv(model, X, y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=kfolds))
    return rmse
xgboost = XGBRegressor(learning_rate=0.01, n_estimators=3460, max_depth=3, min_child_weight=0, gamma=0, subsample=0.7, colsample_bytree=0.7, verbosity=0, objective='reg:squarederror', nthread=-1, scale_pos_weight=1, seed=SEED, reg_alpha=6e-05)