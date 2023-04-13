import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pprint
from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, PredefinedSplit, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.metrics import mean_squared_log_error, make_scorer
from hpsklearn import HyperoptEstimator, svr, svr_linear, svr_rbf, svr_poly, svr_sigmoid, knn_regression, ada_boost_regression, gradient_boosting_regression, random_forest_regression, extra_trees_regression, sgd_regression, xgboost_regression
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 400)
random_state = 42
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
df_target = _input1[['SalePrice']]
df = _input1.append(_input0)
df = df.drop(columns=['Id'], inplace=False)

def mount_df_info(df):
    df_info = df.dtypes.to_frame(name='type')
    df_info['count_null'] = df.isnull().values.sum(axis=0)
    df_info['nunique'] = df.nunique().values
    df_info['count_zeros'] = (df == 0).values.sum(axis=0)
    df_info['max_value'] = df.max()
    df_info['min_value'] = df.min()
    return df_info
df_info = mount_df_info(df)
df_info.sort_values(by='count_null', ascending=False).head()
columns = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
df[columns] = df[columns].fillna('nan')
df_info = mount_df_info(df)
df_info.sort_values(by='count_null', ascending=False).head()
df.loc[df[df['GarageYrBlt'] == 2207].index, 'GarageYrBlt'] = 2010
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0).astype(int)
columns = ['GarageCond', 'GarageFinish', 'GarageQual', 'GarageType']
df[columns] = df[columns].fillna('nan')
df_info = mount_df_info(df)
df_info.sort_values(by='count_null', ascending=False).head()
columns = ['BsmtCond', 'BsmtExposure', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1']
df[columns] = df[columns].fillna('nan')
df_info = mount_df_info(df)
df_info.sort_values(by='count_null', ascending=False).head()
df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
columns = ['Electrical', 'KitchenQual', 'MSZoning', 'Exterior1st', 'Exterior2nd', 'SaleType']
for _column in columns:
    df[_column] = df[_column].fillna(df[_column].mode()[0])
columns = ['MasVnrArea', 'BsmtHalfBath', 'BsmtFullBath', 'GarageArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageCars']
df[columns] = df[columns].fillna(0)
columns = ['MasVnrType', 'Functional', 'Utilities']
df[columns] = df[columns].fillna('nan')
df_info = mount_df_info(df)
df_info.sort_values(by='nunique', ascending=False).head()
columns_disc = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'CentralAir', 'Electrical', 'Functional', 'GarageType', 'GarageFinish', 'PavedDrive', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
columns_disc_qual = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
columns_cont = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
columns_year = ['YearBuilt', 'GarageYrBlt', 'YearRemodAdd', 'YrSold', 'MoSold']
print('Columns disc', len(columns_disc))
print('Columns disc to cont', len(columns_disc_qual))
print('Columns cont', len(columns_cont))
print('Columns year', len(columns_year))
columns_disc = columns_disc + columns_disc_qual
_input0 = df.iloc[_input1.shape[0]:]
_input1 = df.iloc[:_input1.shape[0]]
print(_input1.shape)
print(_input0.shape)
(fig, axs) = plt.subplots(len(columns_cont) // 3 + 1, 3, figsize=(18, 40))
i = 0
corr_with_sales_price = _input1[columns_cont + ['SalePrice']].corr(method='spearman')['SalePrice'].sort_values(ascending=False)
for _column in corr_with_sales_price.index[1:]:
    _ax = axs[i // 3, i % 3]
    _ax.boxplot(_input1[_column], whis=3)
    _ax.set_title(f'{_column} - {corr_with_sales_price[_column]}')
    i += 1
(fig, axs) = plt.subplots(len(columns_disc) // 3 + 1, 3, figsize=(18, 60))
i = 0
for _column in columns_disc:
    _ax = axs[i // 3, i % 3]
    _ax.hist(_input1[_column])
    _ax.set_title(f'{_column}')
    i += 1
df_corr_disc = pd.DataFrame(columns=['feature', 'value', 'corr_with_sale_prices', 'count'])
for _feature in columns_disc:
    df_temp = pd.get_dummies(_input1[_feature])
    for _column in df_temp.columns:
        _corr = stats.pointbiserialr(df_temp[_column], _input1['SalePrice'])
        df_corr_disc = df_corr_disc.append({'feature': _feature, 'value': _column, 'corr_with_sale_prices': _corr[0], 'count': sum(df_temp[_column])}, ignore_index=True)
df_corr_disc.sort_values(by='corr_with_sale_prices')
_input1['new_Fireplaces'] = _input1['Fireplaces'] * _input1['FireplaceQu'].map({'nan': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}).astype(int)
df['new_Fireplaces'] = df['Fireplaces'] * df['FireplaceQu'].map({'nan': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}).astype(int)
_input1[['new_Fireplaces', 'SalePrice']].corr(method='spearman')
df_temp = _input1[_input1['YearRemodAdd'] > _input1['YearBuilt']]
plt.hist(df_temp['YearRemodAdd'] - df_temp['YearBuilt'], bins=30)
_input1['new_has_remod'] = _input1['YearRemodAdd'] > _input1['YearBuilt']
_input1['new_time_remod'] = _input1['YearRemodAdd'] - _input1['YearBuilt']
df['new_has_remod'] = df['YearRemodAdd'] > df['YearBuilt']
df['new_time_remod'] = df['YearRemodAdd'] - df['YearBuilt']
print(stats.pointbiserialr(_input1['new_has_remod'], _input1['SalePrice']))
_input1[['new_time_remod', 'SalePrice']].corr(method='spearman')
plt.hist(_input1['YrSold'] - _input1['YearBuilt'], bins=30)
_input1['new_time_sold'] = _input1['YrSold'] - _input1['YearBuilt']
df['new_time_sold'] = df['YrSold'] - df['YearBuilt']
_input1[['new_time_sold', 'SalePrice']].corr(method='spearman')
_input1['new_garage_after_build'] = _input1['GarageYrBlt'] > _input1['YearBuilt']
df['new_garage_after_build'] = df['GarageYrBlt'] > df['YearBuilt']
print(stats.pointbiserialr(_input1['new_garage_after_build'], _input1['SalePrice']))
_input1['new_totalSF'] = _input1['TotalBsmtSF'] + _input1['1stFlrSF'] + _input1['2ndFlrSF'] + _input1['GarageArea']
df['new_totalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF'] + df['GarageArea']
_input1[['new_totalSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GarageArea', 'SalePrice']].corr(method='spearman')
_input1['BsmtFullBath'] = _input1['BsmtFullBath'].astype(int)
df['BsmtFullBath'] = df['BsmtFullBath'].astype(int)
_input1['BsmtHalfBath'] = _input1['BsmtHalfBath'].astype(int)
df['BsmtHalfBath'] = df['BsmtHalfBath'].astype(int)
_input1['new_others_room'] = _input1['TotRmsAbvGrd'] - _input1['BedroomAbvGr'] - _input1['KitchenAbvGr']
df['new_others_room'] = df['TotRmsAbvGrd'] - df['BedroomAbvGr'] - df['KitchenAbvGr']
_input1['new_all_room'] = _input1['TotRmsAbvGrd'] + _input1['BsmtFullBath'] + _input1['BsmtHalfBath'] + _input1['FullBath'] + _input1['HalfBath']
df['new_all_room'] = df['TotRmsAbvGrd'] + df['BsmtFullBath'] + df['BsmtHalfBath'] + df['FullBath'] + df['HalfBath']
_input1[['TotRmsAbvGrd', 'BedroomAbvGr', 'KitchenAbvGr', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'new_others_room', 'new_all_room', 'SalePrice']].corr(method='spearman')
_df_temp = pd.DataFrame()
for _columns in columns_disc_qual:
    _df_temp[_columns] = df[_columns].map({'nan': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}).astype(int)
df['new_overall'] = _df_temp[columns_disc_qual].sum(axis=1) / len(columns_disc_qual)
df.plot(kind='scatter', x='new_overall', y='OverallQual')
columns_disc = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'CentralAir', 'Electrical', 'Functional', 'GarageType', 'GarageFinish', 'PavedDrive', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition', 'MoSold', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC', 'new_has_remod', 'new_garage_after_build']
columns_cont = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'new_Fireplaces', 'new_time_remod', 'new_time_sold', 'new_totalSF', 'new_others_room', 'new_all_room', 'new_overall']
columns_year = ['YearBuilt', 'GarageYrBlt', 'YearRemodAdd', 'YrSold']
print('Columns disc', len(columns_disc))
print('Columns cont', len(columns_cont))
print('Columns year', len(columns_year))
columns_cont = columns_cont + columns_year
oneHot = OneHotEncoder(handle_unknown='ignore')
df_onehot = oneHot.fit_transform(df[columns_disc])
df_disc = pd.DataFrame.sparse.from_spmatrix(df_onehot, columns=oneHot.get_feature_names(df[columns_disc].columns)).astype(bool).reset_index()
df_cont = df[columns_cont].reset_index()
df_processed = pd.concat([df_cont, df_disc], sort=False, axis=1).drop(columns=['index'])
standard = StandardScaler()
df_processed[columns_cont] = standard.fit_transform(df_processed[columns_cont])
transformer = FunctionTransformer(np.log1p)
df_target = transformer.transform(df_target)
df_processed[columns_cont].hist(figsize=(18, 30))
_input1 = df_processed.iloc[:_input1.shape[0]]
_input0 = df_processed.iloc[_input1.shape[0]:]
print(_input1.shape)
print(_input0.shape)

def rmsqle(y, y_pred):
    return np.sqrt(mean_squared_log_error(np.expm1(np.abs(y)), np.expm1(np.abs(y_pred))))

def evaluate(model, _X_test, _y_test):
    _y_pred = model.predict(_X_test)
    return rmsqle(_y_test, _y_pred)

def write_submission(_result):
    _input2 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
    _input2['SalePrice'] = np.expm1(_result)
    _input2.head()
rmsqle_score = make_scorer(rmsqle, greater_is_better=False)
columns_to_drop = ['MSSubClass_150', 'Utilities_nan', 'Functional_nan']
_input1 = _input1.drop(columns=columns_to_drop, inplace=False)
_input0 = _input0.drop(columns=columns_to_drop, inplace=False)
(X_train, X_test, y_train, y_test) = train_test_split(_input1, df_target, test_size=0.25, random_state=random_state)
from xgboost import XGBRegressor, plot_importance
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
xgb = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, learning_rate=0.05, max_depth=3, min_child_weight=1.7817, n_estimators=2200, reg_alpha=0.464, reg_lambda=0.8571, subsample=0.5213, random_state=7, nthread=-1)