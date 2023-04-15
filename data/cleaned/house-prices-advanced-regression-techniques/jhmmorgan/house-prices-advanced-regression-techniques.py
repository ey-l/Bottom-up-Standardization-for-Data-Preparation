import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import boxcox, inv_boxcox
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_squared_log_error as MSLE

class color:
    PURPLE = '\x1b[95m'
    CYAN = '\x1b[96m'
    DARKCYAN = '\x1b[36m'
    BLUE = '\x1b[94m'
    GREEN = '\x1b[92m'
    YELLOW = '\x1b[93m'
    RED = '\x1b[91m'
    BOLD = '\x1b[1m'
    UNDERLINE = '\x1b[4m'
    HEADER = BOLD + UNDERLINE
    END = '\x1b[0m'
_na_values = ['-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A N/A', '#N/A', 'N/A', 'n/a', '<NA>', '#NA', 'NULL', 'null', 'NaN', '-NaN', 'nan', '-nan', '']
src_path = '_data/input/house-prices-advanced-regression-techniques/'
test_df = pd.read_csv(src_path + 'test.csv', keep_default_na=True)
train_df = pd.read_csv(src_path + 'train.csv', keep_default_na=True)
sample_sub_df = pd.read_csv(src_path + 'sample_submission.csv')
train_df.head()
for df in [train_df, test_df]:
    df.set_index('Id', inplace=True)
train_df.head()
df = pd.DataFrame({'Column': train_df.columns, 'Dtype': train_df.dtypes.astype('str').tolist(), 'Sample1': train_df.loc[1].tolist(), 'Sample2': train_df.loc[50].tolist(), 'Sample3': train_df.loc[500].tolist()})
print(color.BOLD + color.UNDERLINE + 'Data Types for all features in the training data frame' + color.END)
print(df.to_string())

def fix_category_NA(df):
    features_with_NA = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']
    for feature in features_with_NA:
        df.replace({feature: {np.NAN: 'NA'}}, inplace=True)
for df in [train_df, test_df]:
    fix_category_NA(df)
print(color.HEADER + 'Head of some of the columns we amended' + color.END)
train_df[['Alley', 'BsmtQual', 'GarageType', 'PoolQC', 'MiscFeature']].head()
categorical_features = train_df.select_dtypes(exclude=[np.number, bool]).columns
print(color.HEADER + 'Unique values for each categorical feature' + color.END)
for categories in categorical_features:
    print(color.BOLD + categories + color.END)
    print(pd.concat([train_df, test_df])[categories].sort_values().unique())

def fix_categories_integers(df):
    df.replace({'MSSubClass': {20: 'SC20', 30: 'SC30', 40: 'SC40', 45: 'SC45', 50: 'SC50', 60: 'SC60', 70: 'SC70', 75: 'SC75', 80: 'SC80', 85: 'SC85', 90: 'SC90', 120: 'SC120', 150: 'SC150', 160: 'SC160', 180: 'SC180', 190: 'SC190'}, 'MoSold': {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}, 'CentralAir': {'Y': True, 'N': False}}, inplace=True)
    df['YrSold'] = pd.Categorical(df.YrSold)
for df in [train_df, test_df]:
    fix_categories_integers(df)
print(color.HEADER + "Head of the features we've fixed" + color.END)
train_df[['MSSubClass', 'YrSold', 'MoSold', 'CentralAir']].head()
_train_df = train_df.drop(columns='SalePrice')
combined_df = pd.concat([_train_df, test_df])
print(color.BOLD + 'Which features contain null values?' + color.END)
print(combined_df.isnull().sum()[combined_df.isnull().sum() > 0])

def fix_missing_values(df):
    MSZoning_series = df.groupby('Neighborhood').MSZoning.agg(lambda x: x.value_counts().index[0])
    LotFrontage_series = df.groupby('Neighborhood').LotFrontage.median()
    df.fillna({'Utilities': 'AllPub', 'Exterior1st': 'Other', 'Exterior2nd': 'Other', 'MasVnrType': 'None', 'MasVnrArea': 0, 'BsmtFinSF1': 0, 'BsmtFinSF2': 0, 'BsmtUnfSF': 0, 'TotalBsmtSF': 0, 'Electrical': 'SBrkr', 'BsmtFullBath': 0, 'BsmtHalfBath': 0, 'KitchenQual': 'TA', 'Functional': 'Typ', 'GarageYrBlt': 'None', 'GarageCars': 0, 'GarageArea': 0, 'SaleType': 'Oth', 'MSZoning': df['Neighborhood'].apply(lambda x: MSZoning_series[x]), 'LotFrontage': df['Neighborhood'].apply(lambda x: LotFrontage_series[x])}, inplace=True)
    df['GarageYrBlt'] = pd.Categorical(df.GarageYrBlt)
for df in [train_df, test_df]:
    fix_missing_values(df)
print(color.BOLD + color.RED + 'Number of null values across both train and test data frames?')
print(f"{pd.concat([train_df.drop(['SalePrice'], axis=1), test_df]).isnull().sum().sum()}" + color.END)
plt.figure(figsize=(12, 4))
plt.suptitle('Visualising the skewness of the SalePrice target variable')
plt.subplot(1, 2, 1)
sns.histplot(train_df['SalePrice'], stat='density', kde=True)
plt.title('Distribution Plot')
plt.subplot(1, 2, 2)
stats.probplot(train_df['SalePrice'], plot=plt)
plt.tight_layout()

plt.clf()
train_df['SalePrice'] = np.log1p(train_df.SalePrice)
plt.figure(figsize=(12, 4))
plt.suptitle('Visualisaing the skewnewss of the SalePrice target variable following a log1p transformation')
plt.subplot(1, 2, 1)
sns.histplot(train_df['SalePrice'], stat='density', kde=True)
plt.title('Distribution Plot')
plt.subplot(1, 2, 2)
stats.probplot(train_df['SalePrice'], plot=plt)
plt.tight_layout()

plt.clf()
numerical_features = train_df.select_dtypes(include=[np.number]).columns
print(color.BOLD + f'Numerical Features ({len(numerical_features)}):' + color.END + f'\n{numerical_features}')
categorical_features = train_df.select_dtypes(exclude=[np.number, bool]).columns
print(color.BOLD + f'Categorical Features ({len(categorical_features)}):' + color.END + f'\n{categorical_features}')
numerical_groups = math.ceil(len(numerical_features.values) / 8)
categorical_groups = math.ceil(len(categorical_features.values) / 8)
total_groups = numerical_groups + categorical_groups
numerical_step = 8
categorical_step = 8
group_num = np.empty(int(numerical_groups), dtype=pd.Series)
for grp in np.arange(numerical_groups):
    st = int(grp * numerical_step)
    en = int((grp + 1) * numerical_step - 1) + 1
    group_num[int(grp)] = numerical_features[st:en]
group_cat = np.empty(int(categorical_groups), dtype=pd.Series)
for grp in np.arange(categorical_groups):
    st = int(grp * categorical_step)
    en = int((grp + 1) * categorical_step - 1) + 1
    group_cat[int(grp)] = categorical_features[st:en]
print(color.BOLD + color.UNDERLINE + 'Visualisation of distribution and relationship of numerical features vs SalePrice' + color.END)
groups = group_num
for grp in groups:
    plt.figure(figsize=(12, 12))
    i = 1
    for feature in grp:
        width = 4
        height = 4
        _ = plt.subplot(height, width, i)
        _ = sns.histplot(train_df[feature], kde=True, stat='density', linewidth=0)
        _ = plt.title('Distribution')
        i += 1
        _ = plt.subplot(height, width, i)
        _ = sns.scatterplot(data=train_df, x=feature, y='SalePrice', alpha=0.5)
        _ = plt.title('Relationship')
        i += 1
    plt.tight_layout()

    plt.clf()
old_length = len(train_df)
train_df = train_df.drop(train_df[(train_df.LotFrontage > 200) | (train_df.LotArea > 100000) | (train_df.BsmtFinSF1 > 4000) | (train_df.BsmtFinSF2 > 1200) | (train_df.TotalBsmtSF > 5000) | (train_df.GrLivArea > 4000) | (train_df.KitchenAbvGr == 0) | (train_df.WoodDeckSF > 750) | (train_df.OpenPorchSF > 500) | (train_df.EnclosedPorch > 500) | (train_df.MiscVal > 5000)].index)
new_length = len(train_df)
print(color.HEADER + color.RED + f'Reduction in training data from removing outliers is {np.round(100 * (old_length - new_length) / old_length, 2)}%' + color.END)

def numerical_feature_engineering(df):
    df['Has_LowQualFinSF'] = df['LowQualFinSF'].apply(lambda x: False if x == 0 else True)
    df['Has_Pool'] = df['PoolArea'].apply(lambda x: False if x == 0 else True)
    df['LivAreaRatio'] = df.GrLivArea / df.LotArea
    df['SpaceRatio'] = (df['1stFlrSF'] + df['2ndFlrSF']) / df['TotRmsAbvGrd']
    df['TotalBath'] = df.BsmtFullBath + df.BsmtHalfBath
    df['TotalRoom'] = df.TotRmsAbvGrd + df.FullBath + df.HalfBath
    df['BsmtFullBath'] = df['BsmtFullBath'].apply(lambda x: False if x == 0 else True)
    df['BsmtHalfBath'] = df['BsmtHalfBath'].apply(lambda x: False if x == 0 else True)
    df['HalfBath'] = df['HalfBath'].apply(lambda x: False if x == 0 else True)
    df['BedroomAbvGr'] = df['BedroomAbvGr'].apply(lambda x: x if x < 5 else 5)
    df['KitchenAbvGr'] = df['KitchenAbvGr'].apply(lambda x: x if x < 2 else 2)
    df['Fireplaces'] = df['Fireplaces'].apply(lambda x: x if x < 2 else 2)
    df['GarageCars'] = df['GarageCars'].apply(lambda x: x if x < 3 else 3)
    df['NhbdRank'] = df.groupby('Neighborhood')['GrLivArea'].transform('median')
    df['GrLivAreaPlusBsmtSF'] = df.GrLivArea + df.TotalBsmtSF
    df['RecentRemodLargeBsmt'] = df.YearRemodAdd * df.TotalBsmtSF
    df.drop(columns=['LowQualFinSF', 'PoolArea'], inplace=True)
for df in [train_df, test_df]:
    numerical_feature_engineering(df)
print(color.HEADER + 'Shape of our training dataframe, after numerical feature engineering' + color.END)
train_df.shape
print(color.HEADER + 'Visualisation of distribution and relationship of categorical features vs SalePrice' + color.END)
groups = group_cat
for grp in groups:
    plt.figure(figsize=(12, 12))
    i = 1
    for feature in grp:
        width = 4
        height = 4
        _ = plt.subplot(height, width, i)
        _ = sns.countplot(x=train_df[feature])
        _ = plt.xticks(rotation=90)
        _ = plt.title('Distribution')
        i += 1
        _ = plt.subplot(height, width, i)
        _ = sns.stripplot(data=train_df, x=feature, y='SalePrice', alpha=0.5)
        _ = plt.xticks(rotation=90)
        _ = plt.title('Relationship')
        i += 1
    plt.tight_layout()

    plt.clf()

def categorical_feature_engineering(df):
    df['RoofMatl'] = df['RoofMatl'].apply(lambda x: x if x == 'CompShg' else 'Other')
    df['ExterQual'] = df['ExterQual'].apply(lambda x: 'Good' if x in ['Gd', 'Ex'] else 'Average')
    df['Heating'] = df['Heating'].apply(lambda x: x if x == 'GasA' else 'Other')
    df['Electrical'] = df['Electrical'].apply(lambda x: x if x == 'SBrkr' else 'Other')
    df['KitchenQual'] = df['KitchenQual'].apply(lambda x: 'Good' if x in ['Gd', 'Ex'] else 'Average')
    df['Functional'] = df['Functional'].apply(lambda x: x if x == 'Typ' else 'Other')
    df['SaleType'] = df['SaleType'].apply(lambda x: x if x in ['WD', 'New'] else 'Other')
    df['FrontageType'] = df[['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']].gt(0.0).sum(axis=1)
    df.drop(columns=['Street', 'Utilities', 'Condition2'], inplace=True)
for df in [train_df, test_df]:
    categorical_feature_engineering(df)
print(color.HEADER + "Head of the categorical features we've amended / added" + color.END)
corr = train_df.drop(columns=['SalePrice']).corr()
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
(f, ax) = plt.subplots(figsize=(22, 12))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
_ = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.3, center=0, square=True, linewidths=0.5, annot=False).set(title='Correlation of features')
_ = plt.xlabel('Feature')
_ = plt.ylabel('Feature')

_ = plt.clf()
test_df_pre = test_df.copy()
train_df_pre = train_df.copy()
_train_df = train_df.drop(columns='SalePrice')
categorical_features = pd.concat([_train_df, test_df_pre]).select_dtypes(exclude=[np.number, bool]).columns
combined_df_cat = pd.concat([_train_df, test_df_pre])[categorical_features].reset_index(drop=True)
encoder_mapping = pd.DataFrame(index=categorical_features, columns={'encoder', 'mapping'})
for i in np.arange(len(categorical_features)):
    le = LabelEncoder()