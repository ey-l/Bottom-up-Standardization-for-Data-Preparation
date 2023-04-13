import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from sklearn.preprocessing import OrdinalEncoder
from category_encoders import MEstimateEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate, GridSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.linear_model import ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
train_null = _input1.isna().sum()
test_null = _input0.isna().sum()
missing = pd.DataFrame(data=[train_null, train_null / _input1.shape[0] * 100, test_null, test_null / _input0.shape[0] * 100], columns=_input1.columns, index=['Train Null', 'Train Null (%)', 'Test Null', 'Test Null (%)']).T.sort_values(['Train Null', 'Test Null'], ascending=False)
missing = missing.loc[(missing['Train Null'] > 0) | (missing['Test Null'] > 0)]
missing.style.background_gradient('summer_r')
missing.loc[missing['Train Null (%)'] > 5, ['Train Null (%)', 'Test Null (%)']].iloc[::-1].plot.barh(figsize=(8, 6))
plt.title('Variables With More Than 5% Missing Values')
df_missing = _input1[missing.index]
missing_cat = df_missing.loc[:, df_missing.dtypes == 'object'].columns
missing_num = df_missing.loc[:, df_missing.dtypes != 'object'].columns
print(f'number of categorical variables with missing values: {len(missing_cat)}')
print(f'number of numerical variables with missing values: {len(missing_num)}')
(fig, ax) = plt.subplots(6, 4, figsize=(20, 24))
ax = ax.flatten()
for (i, var) in enumerate(missing_cat):
    count = sns.countplot(data=_input1, x=var, ax=ax[i])
    for bar in count.patches:
        count.annotate(format(bar.get_height()), (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha='center', va='center', size=11, xytext=(0, 8), textcoords='offset points')
    ax[i].set_title(f'{var} Distribution')
    if _input1[var].nunique() > 6:
        ax[i].tick_params(axis='x', rotation=45)
plt.subplots_adjust(hspace=0.5)
(fig, ax) = plt.subplots(3, 4, figsize=(20, 10))
ax = ax.flatten()
for (i, var) in enumerate(missing_num):
    sns.histplot(data=_input1, x=var, kde=True, ax=ax[i])
    ax[i].set_title(f'{var} Distribution')
plt.subplots_adjust(hspace=0.5)
cat_none_var = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'GarageType']
cat_nb_var = ['BsmtExposure', 'BsmtFinType2', 'BsmtCond', 'BsmtQual', 'BsmtFinType1']
cat_mode_var = ['Electrical', 'Functional', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'MSZoning', 'SaleType', 'MasVnrType']
cat_mode = {var: _input1[var].mode()[0] for var in cat_mode_var}
for df in [_input1, _input0]:
    df[cat_none_var] = df[cat_none_var].fillna('None')
    df[cat_nb_var] = df[cat_nb_var].fillna('NB')
    for var in cat_mode_var:
        df[var] = df[var].fillna(cat_mode[var], inplace=False)
    df = df.drop('Utilities', axis=1, inplace=False)
missing_cat = missing_cat.drop('Utilities')
print(f'Categorical variable missing values in train data: {_input1[missing_cat].isna().sum().sum()}')
print(f'Categorical variable missing values in test data: {_input0[missing_cat].isna().sum().sum()}')
mean_LF = _input1.groupby('Street')['LotFrontage'].mean()
num_zero_var = missing_num.drop('LotFrontage')
for df in [_input1, _input0]:
    df.loc[df['LotFrontage'].isna() & (df['Street'] == 'Grvl'), 'LotFrontage'] = mean_LF['Grvl']
    df.loc[df['LotFrontage'].isna() & (df['Street'] == 'Pave'), 'LotFrontage'] = mean_LF['Pave']
    for var in num_zero_var:
        df[var] = df[var].fillna(0, inplace=False)
print(f'Numerical variable missing values in train data: {_input1[missing_num].isna().sum().sum()}')
print(f'Numerical variable missing values in test data: {_input0[missing_num].isna().sum().sum()}')
cat_var = _input1.loc[:, _input1.dtypes == 'object'].nunique()
num_var = _input1.loc[:, _input1.dtypes != 'object'].columns
cat_var_unique = {var: sorted(_input1[var].unique()) for var in cat_var.index}
for (key, val) in cat_var_unique.items():
    cat_var_unique[key] += ['-' for x in range(25 - len(val))]
pd.DataFrame.from_dict(cat_var_unique, orient='index').sort_values([x for x in range(25)])
ord_var1 = ['ExterCond', 'HeatingQC']
ord_var1_cat = ['Po', 'Fa', 'TA', 'Gd', 'Ex']
ord_var2 = ['ExterQual', 'KitchenQual']
ord_var2_cat = ['Fa', 'TA', 'Gd', 'Ex']
ord_var3 = ['FireplaceQu', 'GarageQual', 'GarageCond']
ord_var3_cat = ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
ord_var4 = ['BsmtQual']
ord_var4_cat = ['NB', 'Fa', 'TA', 'Gd', 'Ex']
ord_var5 = ['BsmtCond']
ord_var5_cat = ['NB', 'Po', 'Fa', 'TA', 'Gd']
ord_var6 = ['BsmtExposure']
ord_var6_cat = ['NB', 'No', 'Mn', 'Av', 'Gd']
ord_var7 = ['BsmtFinType1', 'BsmtFinType2']
ord_var7_cat = ['NB', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']
ord_var = [ord_var1, ord_var2, ord_var3, ord_var4, ord_var5, ord_var6, ord_var7]
ord_var_cat = [ord_var1_cat, ord_var2_cat, ord_var3_cat, ord_var4_cat, ord_var5_cat, ord_var6_cat, ord_var7_cat]
ord_all = ord_var1 + ord_var2 + ord_var3 + ord_var4 + ord_var5 + ord_var6 + ord_var7
for i in range(len(ord_var)):
    enc = OrdinalEncoder(categories=[ord_var_cat[i]])
    for var in ord_var[i]:
        _input1[var] = enc.fit_transform(_input1[[var]])
        _input0[var] = enc.transform(_input0[[var]])
_input1[ord_all]
cat_var = cat_var.drop(ord_all)
onehot_var = cat_var[cat_var < 6].index
_input1 = pd.get_dummies(_input1, prefix=onehot_var, columns=onehot_var)
_input0 = pd.get_dummies(_input0, prefix=onehot_var, columns=onehot_var)
add_var = [var for var in _input1.columns if var not in _input0.columns]
for var in add_var:
    if var != 'SalePrice':
        _input0[var] = 0
_input0 = _input0[_input1.columns.drop('SalePrice')]
cat_var = cat_var.drop(onehot_var)
X_train = _input1.drop('SalePrice', axis=1)
y_train = _input1['SalePrice']
te = MEstimateEncoder(cols=_input1[cat_var.index.append(pd.Index(['MoSold']))])
X_train = te.fit_transform(X_train, y_train)
_input0 = te.transform(_input0)
_input1 = pd.concat([X_train, y_train], axis=1)
plt.figure(figsize=(12, 12))
sns.heatmap(_input1[num_var].drop('Id', axis=1).corr())
df_train_corr = _input1[num_var].corr().abs().unstack().sort_values(kind='quicksort', ascending=False).reset_index()
df_train_corr = df_train_corr.rename(columns={'level_0': 'Feature 1', 'level_1': 'Feature 2', 0: 'Correlation Coefficient'}, inplace=False)
df_train_corr = df_train_corr.drop(df_train_corr.iloc[1::2].index, inplace=False)
df_train_corr = df_train_corr.drop(df_train_corr[df_train_corr['Correlation Coefficient'] == 1.0].index)
high_corr = df_train_corr['Correlation Coefficient'] > 0.5
df_train_corr[high_corr].reset_index(drop=True).style.background_gradient('summer_r')
df_train_corr[high_corr].loc[(df_train_corr['Feature 1'] == 'SalePrice') | (df_train_corr['Feature 2'] == 'SalePrice')].reset_index(drop=True).style.background_gradient('summer_r')
for df in [_input1, _input0]:
    df['GarAreaPerCar'] = (df['GarageArea'] / df['GarageCars']).fillna(0)
    df['GrLivAreaPerRoom'] = df['GrLivArea'] / df['TotRmsAbvGrd']
    df['TotalHouseSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalFullBath'] = df['FullBath'] + df['BsmtFullBath']
    df['TotalHalfBath'] = df['HalfBath'] + df['BsmtHalfBath']
    df['InitHouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodHouseAge'] = df['InitHouseAge'] - (df['YrSold'] - df['YearRemodAdd'])
    df['IsRemod'] = (df['YearRemodAdd'] - df['YearBuilt']).apply(lambda x: 1 if x > 0 else 0)
    df['GarageAge'] = (df['YrSold'] - df['GarageYrBlt']).apply(lambda x: 0 if x > 2000 else x)
    df['IsGarage'] = df['GarageYrBlt'].apply(lambda x: 1 if x > 0 else 0)
    df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']
    df['AvgQualCond'] = (df['OverallQual'] + df['OverallCond']) / 2
for df in [_input1, _input0]:
    df = df.drop(['GarageArea', 'GarageCars', 'GrLivArea', 'TotRmsAbvGrd', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BsmtFullBath', 'HalfBath', 'BsmtHalfBath', 'YrSold', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'OverallQual', 'OverallCond'], axis=1, inplace=False)
X_train = _input1.drop(['Id', 'SalePrice'], axis=1)
y_train = _input1.SalePrice
X_test = _input0.drop('Id', axis=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train_log = np.log10(y_train)
regressors = {'XGB Regressor': XGBRegressor(), 'LGBM Regressor': LGBMRegressor(), 'Lasso': Lasso(), 'Ridge': Ridge(), 'Elastic Net': ElasticNet(), 'Bayesian Ridge': BayesianRidge(), 'SVR': SVR(), 'GB Regressor': GradientBoostingRegressor(random_state=0)}
results = pd.DataFrame(columns=['Regressor', 'Avg_RMSE'])
for (name, reg) in regressors.items():
    model = reg
    cv_results = cross_validate(model, X_train_scaled, y_train_log, cv=10, scoring=['neg_root_mean_squared_error'])
    results = results.append({'Regressor': name, 'Avg_RMSE': np.abs(cv_results['test_neg_root_mean_squared_error']).mean()}, ignore_index=True)
results = results.sort_values('Avg_RMSE', ascending=True)
results.reset_index(drop=True)
plt.figure(figsize=(12, 6))
sns.barplot(data=results, x='Avg_RMSE', y='Regressor')
plt.title('Average RMSE CV Score')
gbr = GradientBoostingRegressor(random_state=0)
params = {'loss': ('squared_error', 'absolute_error'), 'learning_rate': (1.0, 0.1, 0.01), 'n_estimators': (50, 100, 200)}
reg1 = GridSearchCV(gbr, params, cv=10)