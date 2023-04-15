import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
import sklearn.metrics as skm
import lightgbm as lgb
from lightgbm import early_stopping
from sklearn.linear_model import Lasso
random_seed = 12345
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
sns.set_style('dark')
TRAIN_PATH = '_data/input/house-prices-advanced-regression-techniques/train.csv'
TEST_PATH = '_data/input/house-prices-advanced-regression-techniques/test.csv'
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)
df = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)
df
TARGET = 'SalePrice'
plt.figure()
df[TARGET].hist(bins=20)
plt.title(TARGET + ' before transformation')

df[TARGET] = np.log(df[TARGET])
df['z_score_target'] = np.abs(stats.zscore(df[TARGET]))
df = df.loc[df['z_score_target'] < 3].reset_index(drop=True)
del df['z_score_target']
plt.figure()
df[TARGET].hist(bins=20)
plt.title(TARGET + ' after transformation')

NOMINAL_FEATURES = ['BedroomAbvGr', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtQual', 'ExterCond', 'ExterQual', 'Fireplaces', 'FireplaceQu', 'Functional', 'FullBath', 'GarageCars', 'GarageCond', 'GarageQual', 'HalfBath', 'HeatingQC', 'KitchenAbvGr', 'KitchenQual', 'LandSlope', 'LotShape', 'PavedDrive', 'PoolQC', 'Street', 'Utilities', 'OverallCond', 'OverallQual', 'TotRmsAbvGrd']
CATEGORICAL_FEATURES = ['Alley', 'MSSubClass', 'MoSold', 'MSZoning', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'CentralAir', 'Electrical', 'GarageType', 'GarageFinish', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
NUMERICAL_FEATURES = ['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'YrSold']
NUM_FEATURES = NOMINAL_FEATURES + NUMERICAL_FEATURES
ALL_FEATURES = NOMINAL_FEATURES + NUMERICAL_FEATURES + CATEGORICAL_FEATURES
df = df[ALL_FEATURES + [TARGET]].copy()

def fix_missing(df_data):
    df_data.loc[:, 'Alley'] = df_data.loc[:, 'Alley'].fillna('None')
    df_data.loc[:, 'BedroomAbvGr'] = df_data.loc[:, 'BedroomAbvGr'].fillna(0)
    df_data.loc[:, 'BsmtQual'] = df_data.loc[:, 'BsmtQual'].fillna('No')
    df_data.loc[:, 'BsmtCond'] = df_data.loc[:, 'BsmtCond'].fillna('No')
    df_data.loc[:, 'BsmtExposure'] = df_data.loc[:, 'BsmtExposure'].fillna('No')
    df_data.loc[:, 'BsmtFinType1'] = df_data.loc[:, 'BsmtFinType1'].fillna('No')
    df_data.loc[:, 'BsmtFinType2'] = df_data.loc[:, 'BsmtFinType2'].fillna('No')
    df_data.loc[:, 'BsmtFullBath'] = df_data.loc[:, 'BsmtFullBath'].fillna(0)
    df_data.loc[:, 'BsmtHalfBath'] = df_data.loc[:, 'BsmtHalfBath'].fillna(0)
    df_data.loc[:, 'BsmtUnfSF'] = df_data.loc[:, 'BsmtUnfSF'].fillna(0)
    df_data.loc[:, 'TotalBsmtSF'] = df_data.loc[:, 'TotalBsmtSF'].fillna(0)
    df_data.loc[:, 'BsmtFinSF1'] = df_data.loc[:, 'BsmtFinSF1'].fillna(0)
    df_data.loc[:, 'BsmtFinSF2'] = df_data.loc[:, 'BsmtFinSF2'].fillna(0)
    df_data.loc[:, 'CentralAir'] = df_data.loc[:, 'CentralAir'].fillna('N')
    df_data.loc[:, 'Condition1'] = df_data.loc[:, 'Condition1'].fillna('Norm')
    df_data.loc[:, 'Condition2'] = df_data.loc[:, 'Condition2'].fillna('Norm')
    df_data.loc[:, 'EnclosedPorch'] = df_data.loc[:, 'EnclosedPorch'].fillna(0)
    df_data.loc[:, 'ExterCond'] = df_data.loc[:, 'ExterCond'].fillna('TA')
    df_data.loc[:, 'ExterQual'] = df_data.loc[:, 'ExterQual'].fillna('TA')
    df_data.loc[:, 'Fence'] = df_data.loc[:, 'Fence'].fillna('No')
    df_data.loc[:, 'FireplaceQu'] = df_data.loc[:, 'FireplaceQu'].fillna('No')
    df_data.loc[:, 'Fireplaces'] = df_data.loc[:, 'Fireplaces'].fillna(0)
    df_data.loc[:, 'Functional'] = df_data.loc[:, 'Functional'].fillna('Typ')
    df_data.loc[:, 'GarageType'] = df_data.loc[:, 'GarageType'].fillna('No')
    df_data.loc[:, 'GarageFinish'] = df_data.loc[:, 'GarageFinish'].fillna('No')
    df_data.loc[:, 'GarageQual'] = df_data.loc[:, 'GarageQual'].fillna('No')
    df_data.loc[:, 'GarageCond'] = df_data.loc[:, 'GarageCond'].fillna('No')
    df_data.loc[:, 'GarageArea'] = df_data.loc[:, 'GarageArea'].fillna(0)
    df_data.loc[:, 'GarageCars'] = df_data.loc[:, 'GarageCars'].fillna(0)
    df_data.loc[:, 'HalfBath'] = df_data.loc[:, 'HalfBath'].fillna(0)
    df_data.loc[:, 'HeatingQC'] = df_data.loc[:, 'HeatingQC'].fillna('TA')
    df_data.loc[:, 'KitchenAbvGr'] = df_data.loc[:, 'KitchenAbvGr'].fillna(0)
    df_data.loc[:, 'KitchenQual'] = df_data.loc[:, 'KitchenQual'].fillna('TA')
    df_data.loc[:, 'LotFrontage'] = df_data.loc[:, 'LotFrontage'].fillna(0)
    df_data.loc[:, 'LotShape'] = df_data.loc[:, 'LotShape'].fillna('Reg')
    df_data.loc[:, 'MasVnrType'] = df_data.loc[:, 'MasVnrType'].fillna('None')
    df_data.loc[:, 'MasVnrArea'] = df_data.loc[:, 'MasVnrArea'].fillna(0)
    df_data.loc[:, 'MiscFeature'] = df_data.loc[:, 'MiscFeature'].fillna('No')
    df_data.loc[:, 'MiscVal'] = df_data.loc[:, 'MiscVal'].fillna(0)
    df_data.loc[:, 'OpenPorchSF'] = df_data.loc[:, 'OpenPorchSF'].fillna(0)
    df_data.loc[:, 'PavedDrive'] = df_data.loc[:, 'PavedDrive'].fillna('N')
    df_data.loc[:, 'PoolQC'] = df_data.loc[:, 'PoolQC'].fillna('No')
    df_data.loc[:, 'PoolArea'] = df_data.loc[:, 'PoolArea'].fillna(0)
    df_data.loc[:, 'SaleCondition'] = df_data.loc[:, 'SaleCondition'].fillna('Normal')
    df_data.loc[:, 'ScreenPorch'] = df_data.loc[:, 'ScreenPorch'].fillna(0)
    df_data.loc[:, 'TotRmsAbvGrd'] = df_data.loc[:, 'TotRmsAbvGrd'].fillna(0)
    df_data.loc[:, 'Utilities'] = df_data.loc[:, 'Utilities'].fillna('AllPub')
    df_data.loc[:, 'WoodDeckSF'] = df_data.loc[:, 'WoodDeckSF'].fillna(0)
    return df_data
df = fix_missing(df.copy())
df_test = fix_missing(df_test.copy())

def fix_categories(df_data):
    df_data = df_data.replace({'MSSubClass': {20: 'SC20', 30: 'SC30', 40: 'SC40', 45: 'SC45', 50: 'SC50', 60: 'SC60', 70: 'SC70', 75: 'SC75', 80: 'SC80', 85: 'SC85', 90: 'SC90', 120: 'SC120', 150: 'SC150', 160: 'SC160', 180: 'SC180', 190: 'SC190'}, 'MoSold': {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}})
    df_data = df_data.replace({'BsmtCond': {'No': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'BsmtExposure': {'No': 0, 'Mn': 1, 'Av': 2, 'Gd': 3}, 'BsmtFinType1': {'No': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}, 'BsmtFinType2': {'No': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}, 'BsmtQual': {'No': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'ExterCond': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'ExterQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'FireplaceQu': {'No': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'Functional': {'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8}, 'GarageCond': {'No': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'GarageQual': {'No': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'HeatingQC': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'KitchenQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'LandSlope': {'Sev': 1, 'Mod': 2, 'Gtl': 3}, 'LotShape': {'IR3': 1, 'IR2': 2, 'IR1': 3, 'Reg': 4}, 'PavedDrive': {'N': 0, 'P': 1, 'Y': 2}, 'PoolQC': {'No': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}, 'Street': {'Grvl': 1, 'Pave': 2}, 'Utilities': {'ELO': 1, 'NoSeWa': 2, 'NoSewr': 3, 'AllPub': 4}})
    return df_data
df = fix_categories(df.copy())
df_test = fix_categories(df_test.copy())
for col in CATEGORICAL_FEATURES:
    temp = df.groupby(col)[TARGET].count() / len(df)
    temp = temp[temp < 0.02].index
    df[col] = np.where(df[col].isin(temp), 'rare', df[col])
    df_test[col] = np.where(df_test[col].isin(temp), 'rare', df_test[col])
for feature in CATEGORICAL_FEATURES:
    labels_ordered = list(df.groupby([feature])['SalePrice'].mean().sort_values().index)
    if 'missing' not in labels_ordered:
        labels_ordered.append('missing')
    labels_ordered = {k: i for (i, k) in enumerate(labels_ordered, 0)}
    df[feature] = df[feature].map(labels_ordered)
    df_test[feature] = df_test[feature].map(labels_ordered)
df.plot.scatter(x='GrLivArea', y=TARGET)
df = df[df['GrLivArea'] < 4000].reset_index(drop=True)
(fig, ax) = plt.subplots(int(np.ceil(len(NUMERICAL_FEATURES) / 5)), 5, figsize=(25, len(NUMERICAL_FEATURES)))
for (i, col) in enumerate(NUMERICAL_FEATURES):
    df[col].hist(bins=20, ax=ax[i // 5, i % 5])
    ax[i // 5, i % 5].title.set_text(col)
for col in NUMERICAL_FEATURES:
    if col in ['YearBuilt', 'YearRemodAdd', 'YrSold']:
        continue
    skew_before = df[col].skew()
    if (df[col] <= 0).sum() > 0:
        if df.loc[df[col] > 0, col].skew() < 0.5:
            continue
        df['has_zero_' + col] = 0
        df.loc[df[col] > 0, 'has_zero_' + col] = 1
        df.loc[df[col] > 0, col] = np.log(df.loc[df[col] > 0, col])
        print('{} skewness before: {:.2f} skewness after: {:.2f}'.format(col, skew_before, df.loc[df[col] > 0, col].skew()))
        df_test['has_zero_' + col] = 0
        df_test.loc[df_test[col] > 0, 'has_zero_' + col] = 1
        df_test.loc[df_test[col] > 0, col] = np.log(df_test.loc[df_test[col] > 0, col])
        ALL_FEATURES.append('has_zero_' + col)
    else:
        if df[col].skew() < 0.5:
            continue
        df[col] = np.log(df[col])
        df_test[col] = np.log(df_test[col])
        print('{} skewness before: {:.2f} skewness after: {:.2f}'.format(col, skew_before, df[col].skew()))
scalar = RobustScaler()
df[NUM_FEATURES] = scalar.fit_transform(df[NUM_FEATURES])
df_test[NUM_FEATURES] = scalar.transform(df_test[NUM_FEATURES])

def calc_metrics(y, y_pred):
    mae = skm.mean_absolute_error(y, y_pred)
    r2 = skm.r2_score(y, y_pred)
    rmse = np.sqrt(skm.mean_squared_error(y, y_pred))
    corr = np.corrcoef(y, y_pred)[0, 1]
    print_str = 'MAE: {:.2f} R2: {:.2f} RMSE: {:.2f} Corr: {:.2f}'.format(mae, r2, rmse, corr)
    return ({'mae': mae, 'r2': r2, 'rmse': rmse, 'corr': corr}, print_str)

def cv_train_and_evaluate(model, param_grid, x_train, y_train, x_val, y_val, model_name, n_folds=5, scoring='neg_median_absolute_error', fit_params={}):
    clf = GridSearchCV(model, param_grid, cv=n_folds, scoring=scoring, refit=True, verbose=0)