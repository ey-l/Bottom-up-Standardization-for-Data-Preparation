import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import re
import seaborn as sns
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import learning_curve, GridSearchCV, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, Normalizer, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVR
pd.options.display.float_format = '{:,.2f}'.format
sns.set_theme(style='whitegrid', palette='colorblind')
prefix = '_data/input/house-prices-advanced-regression-techniques/'
train_df = pd.read_csv(prefix + 'train.csv')
train_df.set_index('Id', inplace=True)
train_df.head()
test_df = pd.read_csv(prefix + 'test.csv')
test_df.set_index('Id', inplace=True)
test_df.head()
train_df.shape
print(train_df.info())

def get_num_cols(df):
    return df.columns[df.dtypes != 'O']
num_cols = get_num_cols(train_df)
num_cols

def get_cat_cols(df):
    return df.columns[df.dtypes == 'O']
cat_cols = get_cat_cols(train_df)
cat_cols
cat_ordered_cols = pd.Index(['LotShape', 'LandSlope', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence'])
cat_ordered_cols
train_df[train_df.isna().all(axis=1)]
nan_count = train_df.isna().sum()
missing_cols = nan_count[nan_count != 0]
missing_cols.sort_values(ascending=False)
test_nan_count = test_df.isna().sum()
test_missing_cols = test_nan_count[test_nan_count != 0]
test_missing_cols.sort_values(ascending=False)
missing_values_summary = pd.concat([missing_cols, test_missing_cols], axis=1)
missing_values_summary.rename(columns={0: 'train', 1: 'test'}, inplace=True)
missing_values_summary.fillna(0, inplace=True)
missing_values_summary.sort_values(by='train', ascending=False)
pure_missing_cols = pd.Index(['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'Electrical', 'SaleType', 'KitchenQual', 'Functional', 'LotFrontage'])
to_fill_mode = {x: train_df[x].mode()[0] for x in pure_missing_cols.join(cat_cols, how='inner')}
to_fill_mean = {x: train_df[x].mean() for x in pure_missing_cols.join(num_cols, how='inner')}
to_fill_none = {x: 'None' for x in ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'MasVnrType', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']}
to_fill_zero = {x: 0.0 for x in ['GarageCars', 'GarageArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'BsmtFullBath', 'BsmtHalfBath', 'TotalBsmtSF', 'MasVnrArea']}
to_fill_other = {'GarageYrBlt': 2010}
to_fill = {**to_fill_mode, **to_fill_mean, **to_fill_none, **to_fill_zero, **to_fill_other}
len(to_fill)
to_fill
set(missing_values_summary.index.to_list()).difference(set(to_fill.keys()))
train_df.fillna(value=to_fill, inplace=True)
test_df.fillna(value=to_fill, inplace=True)
nan_count = train_df.isna().sum()
nan_count = nan_count[nan_count > 0]
nan_count
nan_count = test_df.isna().sum()
nan_count = nan_count[nan_count > 0]
nan_count
train_df[['YrSold', 'YearRemodAdd', 'YearBuilt', 'GarageYrBlt']].describe()
years_df = test_df[['YrSold', 'YearRemodAdd', 'YearBuilt', 'GarageYrBlt']]
years_df.describe()
weird_year = test_df[(years_df > 2010).any(axis=1)][['YrSold', 'YearRemodAdd', 'YearBuilt', 'GarageYrBlt']]
weird_year
test_df.loc[weird_year.index, 'GarageYrBlt'] = train_df['GarageYrBlt'].median()
quality_scale = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
quality_scale_with_none = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 3}
cat_ordered_col_replace_values = {'LandSlope': {'Gtl': 2, 'Mod': 3, 'Sev': 0}, 'LotShape': {'Reg': 4, 'IR1': 3, 'IR2': 2, 'IR3': 1, 'None': 0}, 'ExterQual': quality_scale, 'ExterCond': quality_scale, 'BsmtQual': quality_scale_with_none, 'BsmtCond': quality_scale_with_none, 'BsmtExposure': {'Gd': 3, 'Av': 2, 'Mn': 1, 'No': 0, 'None': 1.5}, 'BsmtFinType1': {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'None': 0}, 'BsmtFinType2': {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'None': 0}, 'Functional': {'Typ': 7, 'Min1': 6, 'Min2': 5, 'Mod': 4, 'Maj1': 3, 'Maj2': 2, 'Sev': 1, 'Sal': 0}, 'HeatingQC': quality_scale, 'KitchenQual': quality_scale, 'FireplaceQu': quality_scale_with_none, 'GarageFinish': {'Fin': 3, 'RFn': 2, 'Unf': 1, 'None': 0}, 'GarageQual': quality_scale, 'GarageCond': quality_scale_with_none, 'PoolQC': quality_scale_with_none, 'Fence': {'GdPrv': 3, 'MnPrv': 2, 'GdWo': 1, 'MnWw': 0, 'None': 1.5}}
train_df.replace(cat_ordered_col_replace_values, inplace=True)
test_df.replace(cat_ordered_col_replace_values, inplace=True)
train_df[cat_ordered_col_replace_values.keys()].dtypes
test_df[cat_ordered_col_replace_values.keys()].dtypes
features_with_outliers = ['LotFrontage', 'LotArea', 'MasVnrArea', 'GrLivArea', 'TotalBsmtSF', 'SalePrice']
sns.pairplot(train_df[features_with_outliers])
(fig, ax) = plt.subplots(figsize=(10, 8))
ax.set_xticks(ax.get_xticks(), minor=True)
sns.regplot(x=train_df['GrLivArea'], y=train_df['SalePrice'], ax=ax)
(fig, axes) = plt.subplots(3, 2, figsize=(15, 12))
for (i, col) in enumerate(features_with_outliers):
    sns.histplot(train_df[col], kde=True, ax=axes[i // 2][i % 2])
length = train_df.shape[0]
thresholds = pd.DataFrame(data={'LotFrontage': [170], 'LotArea': [20000], 'MasVnrArea': [1500], 'GrLivArea': [3500], 'TotalBsmtSF': [4000], 'SalePrice': [550000]})
thresholds = thresholds.loc[thresholds.index.repeat(length)]
thresholds.index = train_df.index
outliers = (train_df[['LotFrontage', 'LotArea', 'MasVnrArea', 'GrLivArea', 'TotalBsmtSF', 'SalePrice']] >= thresholds).any(axis=1)
train_df = train_df.loc[~outliers]
sns.pairplot(train_df[features_with_outliers])
(fig, axes) = plt.subplots(1, 2, figsize=(15, 6))
sns.boxplot(data=train_df, x='YearBuilt', ax=axes[0])
sns.boxplot(data=train_df, x='YearRemodAdd', ax=axes[1])
train_df['Remod'] = (train_df['YearBuilt'] == train_df['YearRemodAdd']).astype(int)
sns.pairplot(data=train_df[['YearBuilt', 'Remod', 'YearRemodAdd', 'SalePrice']])

def group_then_agg(df, column, target='SalePrice', agg=['mean', 'std', 'count'], sort='mean', price_to_thousand=True):
    grouped = df.groupby(column)[target].agg(agg).sort_values(by=sort, ascending=False)
    if price_to_thousand:
        grouped['mean'] = grouped['mean'] / 1000
        grouped['std'] = grouped['std'] / 1000
    return grouped

def plot_cat_cols(df, columns, max_filter=10):
    num_axes = len(columns)
    num_rows = int(np.ceil(np.sqrt(num_axes)))
    num_cols = int(np.ceil(num_axes / num_rows))

    def get_ax(axes, i):
        if num_axes == 1:
            return axes
        if num_rows == 1 or num_cols == 1:
            return axes[i]
        return axes[i // num_cols][i % num_cols]
    (fig, axes) = plt.subplots(num_rows, num_cols, figsize=(num_cols * 8, num_rows * 8))
    for (i, col) in enumerate(columns):
        data = group_then_agg(df, col)
        data = data.iloc[:max_filter, :]
        data = data[['mean', 'count']].melt(ignore_index=False).reset_index()
        sns.barplot(data=data, x='value', y=col, hue='variable', orient='h', ax=get_ax(axes, i))
plot_cat_cols(train_df, ['Neighborhood', 'Street', 'Alley', 'LotShape', 'LotConfig', 'LandContour', 'Condition1', 'Condition2'])
sns.regplot(data=train_df, x='LotShape', y='SalePrice')
data = train_df[['LotFrontage', 'SalePrice']].copy()
data = StandardScaler().fit_transform(data)
sns.regplot(x=data[:, 0], y=data[:, 1])
sns.regplot(data=train_df, x='LotArea', y='SalePrice')
plot_cat_cols(train_df, ['BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType'])
sns.pairplot(data=train_df[['ExterQual', 'ExterCond', 'MasVnrArea', 'SalePrice']])

def get_story(style: str):
    if style.startswith('S'):
        return 2
    m = re.match('^(\\d+(?:\\.\\d+)?)', style)
    if m == None:
        return np.nan
    return float(m.group(1))
train_df['HouseStory'] = train_df['HouseStyle'].apply(get_story)
group_then_agg(train_df, 'HouseStory')
sns.regplot(data=train_df, x='HouseStory', y='SalePrice')
train_df['MasVnrType'].apply(lambda x: x != 'None').astype(int)
a = train_df[(train_df['MasVnrType'] != 'None') & (train_df['MasVnrArea'] != 0)]['MasVnrArea']
sns.regplot(x=[0 if t == 0 else np.log(t) for t in a], y=train_df[(train_df['MasVnrType'] != 'None') & (train_df['MasVnrArea'] != 0)]['SalePrice'])
sns.scatterplot(x=train_df['MasVnrArea'] / train_df['HouseStory'], y=train_df['SalePrice'])
train_df['TotalBath'] = train_df['FullBath'] + train_df['HalfBath'] * 0.5
train_df['TotalFlrSF'] = train_df['1stFlrSF'] + train_df['2ndFlrSF']
fig = sns.pairplot(data=train_df[['TotalBath', 'GrLivArea', 'TotalFlrSF', 'KitchenAbvGr', 'BedroomAbvGr', 'LowQualFinSF', 'TotRmsAbvGrd', 'SalePrice']])
sns.regplot(data=train_df, x='KitchenQual', y='SalePrice')
(train_df['GrLivArea'] - (train_df['1stFlrSF'] + train_df['2ndFlrSF'] + train_df['LowQualFinSF'])).describe()
train_df['AvgRoomSF'] = train_df['GrLivArea'] / train_df['TotRmsAbvGrd']
sns.regplot(x=train_df['AvgRoomSF'], y=train_df['SalePrice'])
train_df[['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']]
sns.pairplot(train_df[['BsmtFinType1', 'BsmtFinType2', 'SalePrice']])
sns.pairplot(train_df[['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'SalePrice']])
train_df['TotalBsmtBath'] = train_df['BsmtFullBath'] + train_df['BsmtHalfBath'] + 0.5
sns.regplot(x=train_df['TotalBsmtBath'], y=train_df['SalePrice'])
sns.pairplot(data=train_df[['GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'SalePrice']])
group_then_agg(train_df, 'GarageType')
sns.scatterplot(x=train_df['GarageYrBlt'], y=train_df['YearBuilt'])
sns.lineplot(x=np.linspace(1880, 2010, 1000), y=np.linspace(1880, 2010, 1000), color='orange')
train_df[train_df['GarageYrBlt'] < train_df['YearBuilt']]

def area_per_car(garage):
    if garage['GarageCars'] == 0:
        return 0
    return garage['GarageArea'] / garage['GarageCars']
train_df['AreaPerCar'] = train_df[['GarageCars', 'GarageArea']].apply(area_per_car, axis=1)

def car_per_area(garage):
    if garage['GarageArea'] == 0:
        return 0
    return garage['GarageCars'] / (garage['GarageArea'] / 100)
sns.regplot(x=train_df[['GarageCars', 'GarageArea']].apply(car_per_area, axis=1), y=train_df['SalePrice'])
plot_cat_cols(train_df, ['Heating', 'CentralAir', 'Electrical'])
sns.regplot(x=train_df['HeatingQC'], y=train_df['SalePrice'])
group_then_agg(train_df, 'PavedDrive')
sns.pairplot(train_df[['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'SalePrice']])
train_df['DeckArea'] = train_df[['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']].sum(axis=1)
sns.regplot(x=train_df['DeckArea'], y=train_df['SalePrice'])
train_df[['LotArea', '1stFlrSF', 'GarageArea']].describe()
train_df['YardArea'] = train_df['LotArea'] - (train_df['GrLivArea'] + train_df['GarageArea'] * (train_df['GarageType'] == 'Basment'))
sns.regplot(x=train_df['YardArea'], y=train_df['SalePrice'])
plot_cat_cols(train_df, ['Utilities', 'Fence', 'Fireplaces'])
train_df['PoolQC'].value_counts()
sns.pairplot(train_df[['PoolQC', 'PoolArea', 'SalePrice']])
sns.regplot(x=train_df['FireplaceQu'], y=train_df['SalePrice'])
sns.regplot(x=(train_df[['PoolQC', 'Fence', 'MiscFeature']] != 'None').sum(axis=1), y=train_df['SalePrice'])
group_then_agg(train_df, 'MiscFeature')
train_df[train_df['MiscFeature'] != 'None'][['MiscFeature', 'MiscVal']]
sns.pairplot(train_df[['OverallQual', 'OverallCond', 'SalePrice']])
corr = train_df.corr()['SalePrice'].sort_values()
corr[corr > 0]
selected_features = ['YearBuilt', 'LotArea', 'ExterQual', 'MasVnrArea', 'TotalBath', 'GrLivArea', 'KitchenQual', 'TotRmsAbvGrd', 'AvgRoomSF', 'BsmtQual', 'TotalBsmtSF', 'GarageFinish', 'GarageArea', 'HeatingQC', 'DeckArea', 'OverallQual', 'FireplaceQu', 'GarageCars', 'Fireplaces', 'Functional', 'Remod', 'Neighborhood', 'Exterior1st', 'HouseStyle', 'MasVnrType']
X = train_df[selected_features].copy()
y = train_df['SalePrice'].copy()
selected_cat_cols = ['Neighborhood', 'Exterior1st', 'HouseStyle', 'MasVnrType']
selected_num_cols = list(filter(lambda x: x not in selected_cat_cols, selected_features))
preprocessor = ColumnTransformer(transformers=[('scale', StandardScaler(), selected_num_cols), ('onehot', OneHotEncoder(handle_unknown='ignore'), selected_cat_cols)])
linear = Pipeline([('preprocess', preprocessor), ('linear', LinearRegression())])
np.mean(cross_val_score(linear, X, y, scoring='neg_root_mean_squared_error', cv=10))
lasso = Pipeline([('preprocess', preprocessor), ('lasso', Lasso())])
gs_lasso = GridSearchCV(estimator=lasso, param_grid={'lasso__alpha': np.linspace(30, 50, 8)}, scoring='neg_root_mean_squared_error', cv=10)