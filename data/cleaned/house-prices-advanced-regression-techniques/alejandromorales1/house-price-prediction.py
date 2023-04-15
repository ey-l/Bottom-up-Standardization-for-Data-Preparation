import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.stats import norm, skew
import warnings
warnings.filterwarnings('ignore')
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data_dir = Path('_data/input/house-prices-advanced-regression-techniques/')
pd.read_csv(data_dir / 'train.csv').head(2)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.precision', 3)
pd.set_option('display.expand_frame_repr', False)
target_variable = 'SalePrice'
cat_columns = ['MSSubClass', 'category', 'Street', 'Alley', 'LotShape', 'MSZoning', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageQual', 'GarageFinish', 'PavedDrive', 'MiscFeature', 'SaleType', 'SaleCondition', 'ExterQual', 'ExterCond', 'BsmtCond', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence']
date_columns = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']
bool_columns = ['CentralAir']
cat_dict = {c: 'category' for c in cat_columns}
bool_dict = {b: 'bool' for b in bool_columns}
columns_types = cat_dict.update(bool_dict)
train = pd.read_csv(data_dir / 'train.csv', true_values=['Y'], false_values=['N'], dtype=columns_types)
test = pd.read_csv(data_dir / 'test.csv', true_values=['Y'], false_values=['N'], dtype=columns_types)
print(f'train shape: {train.shape}')
print(f'test shape: {test.shape}')
train_ID = train['Id']
test_ID = test['Id']
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)
ntrain = train.shape[0]
ntest = test.shape[0]
Y_train = train['SalePrice'].values
all_df = pd.concat((train, test)).reset_index(drop=True)
all_df.drop(['SalePrice'], axis=1, inplace=True)
print(f'all_df shape: {all_df.shape}')
plt.figure(figsize=(20, 4))
sns.heatmap(all_df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
for cat in all_df.columns[all_df.isnull().any()]:
    if cat in all_df.select_dtypes(include=['category']).columns:
        if 'None' not in all_df[cat].cat.categories:
            all_df[cat] = all_df[cat].cat.add_categories('None')
        all_df[cat] = all_df[cat].fillna('None')
all_df['PoolQC'] = all_df['PoolQC'].fillna('None')
all_df['MiscFeature'] = all_df['MiscFeature'].fillna('None')
all_df['GarageQual'] = all_df['GarageQual'].fillna('None')
all_df['MasVnrArea'] = all_df['MasVnrArea'].fillna(0)
all_df['LotFrontage'] = all_df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
all_df = all_df.drop(['Utilities'], axis=1)
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_df[col] = all_df[col].fillna(0)
plt.figure(figsize=(20, 4))
sns.heatmap(all_df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
sns.distplot(train['SalePrice'])
print(f"Skewness: {train['SalePrice'].skew()}")
print(f"Kurtosis: {train['SalePrice'].kurt()}")
corr = train.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
(f, ax) = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, cmap=cmap, vmax=0.8, square=True, mask=mask)
k = 10
cols = corr.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, cmap=cmap, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

cols = ['OverallQual', 'GarageCars', 'FullBath']
for feature in cols:
    train.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols], size=2.5)

train.plot.scatter(x='GrLivArea', y='SalePrice', color='b')
sns.boxplot(data=train, x='OverallQual', y='SalePrice')
train.plot.scatter(x='1stFlrSF', y='SalePrice', color='b')
saleprice_scaled = (Y_train - Y_train.mean()) / Y_train.std()
saleprice_scaled.sort()
low_range = saleprice_scaled[:10]
high_range = saleprice_scaled[-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)
train.sort_values(by='GrLivArea', ascending=False)[:2]
train = train.drop(train[train_ID == 1299].index)
train = train.drop(train[train_ID == 524].index)
numeric_feats = all_df.dtypes[all_df.dtypes == 'int64'].index
skewed_feats = all_df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print('Skew in numerical features:')
skewness = pd.DataFrame({'Skew': skewed_feats})
skewness.head(10)

def plot_dist(data) -> None:
    fig = plt.figure()
    sns.distplot(data, fit=norm)
    fig = plt.figure()
    res = stats.probplot(data, plot=plt)
plot_dist(Y_train)
plot_dist(np.log1p(Y_train))
from scipy.special import boxcox1p
plot_dist(1 / Y_train)
plot_dist(np.sqrt(Y_train))
plot_dist(Y_train ** (1 / 1.2))
plot_dist(boxcox1p(Y_train, 0.15))
Y_train = np.log1p(Y_train)
plot_dist(np.log(all_df['GrLivArea']))
all_df['GrLivArea'] = np.log(all_df['GrLivArea'])
skewness = skewness[abs(skewness) > 0.75]
skewed_features = skewness.index
for feat in skewed_features:
    if feat is 'GrLivArea':
        continue
    all_df[feat] = boxcox1p(all_df[feat], 0.15)
all_df_dummies = pd.get_dummies(all_df)
all_df_dummies.sample(5)
train_dummies = all_df_dummies[:ntrain]
test_dummies = all_df_dummies[ntrain:]
print(f'train shape: {train_dummies.shape}')
print(f'test shape: {test_dummies.shape}')
print(f'Y_train shape: {Y_train.shape}')
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split

def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn
(X_train, X_test, y_train, y_test) = train_test_split(train_dummies, Y_train, test_size=0.1, random_state=42)
print(f'X_train: {X_train.shape}')
print(f'X_test: {X_test.shape}')
print(f'y_train: {y_train.shape}')
print(f'y_test: {y_test.shape}')

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
D_train = xgb.DMatrix(X_train, label=y_train)
D_test = xgb.DMatrix(X_test, label=y_test)
steps = 20
param = {'colsample_bytree': 0.2, 'max_depth': 4, 'gamma': 0.0, 'learning_rate': 0.01, 'min_child_weight': 1.5, 'n_estimators': 7200, 'reg_alpha': 0.9, 'reg_lambda': 0.6, 'subsample': 0.2, 'seed': 42}
regr = xgb.XGBRegressor(**param)