import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype
from category_encoders import MEstimateEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
import os
sns.set(style='whitegrid', color_codes=True)
sns.set(font_scale=1)
'Ignore deprecation and future, and user warnings.'
import warnings as wrn
wrn.filterwarnings('ignore', category=DeprecationWarning)
wrn.filterwarnings('ignore', category=FutureWarning)
wrn.filterwarnings('ignore', category=UserWarning)
plt.rcParams['figure.figsize'] = (10, 5)
house_attributes = dict()
house_mapping = dict()
file_name = {'train': '_data/input/house-prices-advanced-regression-techniques/train.csv', 'test': '_data/input/house-prices-advanced-regression-techniques/test.csv'}
dict_file_name = '_data/input/house-prices-advanced-regression-techniques/data_description.txt'


def load_dictionary(data_dictory):
    """Read the data dictionary and create a Map"""
    (key, value) = (None, None)
    with open(data_dictory) as file:
        lines = file.readlines()
        for line in lines:
            if len(line.strip()) > 0:
                if len(line[0].strip()) > 0:
                    (key, value) = line.split(':')
                    house_attributes[key.strip()] = value.strip()
                    house_mapping[key] = dict()
                elif key:
                    (attr, *attr_value) = line.strip().split('\t')
                    house_mapping[key][attr] = ' '.join(attr_value)
    return (house_mapping, house_attributes)
(house_mapping, house_attributes) = load_dictionary(dict_file_name)
for key in house_attributes:
    print(f'{key} : {house_attributes[key]}\n    ---------------------------------------------\n    {house_mapping[key]}\n    ')
houseprice = pd.read_csv(file_name['train'], index_col=0)
houseprice_test = pd.read_csv(file_name['test'], index_col=0)
numerical_columns = houseprice.select_dtypes(include=np.number)
categorical_columns = houseprice.select_dtypes(include=['object'])
print(f'Listing the Columns({len(houseprice.columns)} columns):\n\nNumerical columns({len(numerical_columns)}) :  {numerical_columns.columns.tolist()}\n\nCategorical columns({len(categorical_columns)}) :  {categorical_columns.columns.tolist()}\n\n')
print(f'The dataset has data for {houseprice.shape[0]} transaction and columns {houseprice.shape[1]}')
print('Number of dupicated records in the dataset :', len(houseprice[houseprice.duplicated()]))
pd.options.display.float_format = '{:20.2f}'.format
print('Displaying the first 5 records of data:')
houseprice.head(n=5)
stat_saleprice = houseprice.SalePrice.describe()
print(f"Statistics for the SalesPrice:\n---------------------------\ncount    : {stat_saleprice['count']},\nmean     : ${stat_saleprice['mean']},\nstd      : ${stat_saleprice['std']},\nmin      : ${stat_saleprice['min']},\n25%      : ${stat_saleprice['25%']},\n50%      : ${stat_saleprice['50%']},\n75%      : ${stat_saleprice['75%']},\nmax      : ${stat_saleprice['max']},\nIQR      : ${stat_saleprice['25%']} - ${stat_saleprice['75%']}\nskew     : {houseprice.SalePrice.skew()}\nskew(log): {np.log(houseprice.SalePrice).skew()}\nkurt     : {houseprice.SalePrice.kurt()}\nkurt(log): {np.log(houseprice.SalePrice).kurt()}\n")
numerical_columns = houseprice.select_dtypes(include=[np.number]).drop(columns=['SalePrice']).columns.tolist()
uniqueValCount = houseprice[numerical_columns].nunique()
numerical_discrete = uniqueValCount[uniqueValCount < 50].index.tolist()
time_columns = ['YearBuilt', 'YearRemodAdd', 'YrSold', 'MoSold', 'GarageYrBlt']
numerical_discrete = [col for col in numerical_discrete if col not in time_columns]
numerical_columns = [col for col in numerical_columns if col not in numerical_discrete + time_columns]

def data_range_nooutlier(data, level=1, continuous=False, log=False):

    def pct_method(data, level):
        upper = np.percentile(data, 100 - level)
        lower = np.percentile(data, level)
        return [lower, upper]

    def iqr_method(data):
        perc_75 = np.percentile(data, 75)
        perc_25 = np.percentile(data, 25)
        iqr_range = perc_75 - perc_25
        iqr_upper = perc_75 + 1.5 * iqr_range
        iqr_lower = perc_25 - 1.5 * iqr_range
        return [iqr_lower, iqr_upper]

    def std_method(data):
        std = np.std(data)
        upper_3std = np.mean(data) + 3 * std
        lower_3std = np.mean(data) - 3 * std
        return [lower_3std, upper_3std]
    data = data[~data.isna()]
    if log is True:
        data = np.log1p(data)
    pct_range = pct_method(data, level)
    iqr_range = iqr_method(data)
    std_range = std_method(data)
    if continuous is False:
        low_limit = np.min(data)
        high_limit = np.max([pct_range[1], iqr_range[1], std_range[1]])
    elif continuous is True:
        low_limit = np.min([pct_range[0], iqr_range[0], std_range[0]])
        high_limit = np.max([pct_range[1], iqr_range[1], std_range[1]])
    return (low_limit, high_limit)
numerical_columns_summary = houseprice.loc[:, numerical_columns].describe().T
numerical_columns_summary['skew'] = numerical_columns_summary.index.map(lambda x: houseprice[x].skew())
numerical_columns_summary['skew(log1p)'] = numerical_columns_summary.index.map(lambda x: np.log1p(houseprice[x]).skew())
numerical_columns_summary['cutoffmax'] = numerical_columns_summary.index.map(lambda x: data_range_nooutlier(houseprice[x], continuous=True)[1])
numerical_columns_summary
print(f"{len(numerical_columns_summary)} Numerical columns with\n\nMissing Records   : {numerical_columns_summary[numerical_columns_summary['count'] < len(houseprice)].index.tolist()}\nZero units        : {numerical_columns_summary[numerical_columns_summary['min'] == 0].index.tolist()}\nSkew > 5          : {numerical_columns_summary[numerical_columns_summary['skew'] > 5].index.tolist()}\nPossible outliers : {numerical_columns_summary[numerical_columns_summary['max'] > numerical_columns_summary['mean'] + 3 * numerical_columns_summary['std']].index.tolist()}\n")
for col in ['LotArea', 'GrLivArea']:
    print(f"{col} Summary:\n---------------------------\nAverage {col} for the dataset {numerical_columns_summary.loc[col]['mean']} ft^2 with standard deviation {numerical_columns_summary.loc[col]['std']} ft^2\nThe {col} ranges from {numerical_columns_summary.loc[col]['min']} to {numerical_columns_summary.loc[col]['max']} with median of {numerical_columns_summary.loc[col]['50%']}\nCutoff Max for outlier removal: {data_range_nooutlier(houseprice[col], continuous=True)[1]:.2f}\n")
numerical_discrete_summary = houseprice[numerical_discrete].applymap(str).describe().transpose()
numerical_discrete_summary['count'] = numerical_discrete_summary.index.map(lambda x: len(houseprice[~houseprice[x].isna()]))
numerical_discrete_summary['top'] = numerical_discrete_summary.apply(lambda x: house_mapping[x.name][x['top']] if x.name in house_mapping and house_mapping[x.name] else x['top'], axis=1)
numerical_discrete_summary
print(f"{len(numerical_discrete_summary)} Numerical-Discrete columns with\n\nMissing Records   : {numerical_discrete_summary[numerical_discrete_summary['count'] < len(houseprice)].index.tolist()}\n")
for col in ['MSSubClass', 'OverallQual', 'GarageCars']:
    print(f"{col} Summary:\n---------------------------\nNumber for Unique Values  : {numerical_discrete_summary.loc[col, 'unique']}\nValue with most occurances: {numerical_discrete_summary.loc[col, 'top']} ({numerical_discrete_summary.loc[col, 'freq']})\n\n")
time_columns_summary = pd.DataFrame(index=time_columns)
time_columns_summary['count'] = time_columns_summary.index.map(lambda x: len(houseprice[~houseprice[x].isna()])).astype('int')
time_columns_summary['unique'] = time_columns_summary.index.map(lambda x: houseprice[x].nunique()).astype('int')
time_columns_summary['min'] = time_columns_summary.index.map(lambda x: houseprice[x].min()).astype('int')
time_columns_summary['max'] = time_columns_summary.index.map(lambda x: houseprice[x].max()).astype('int')
time_columns_summary['top'] = time_columns_summary.index.map(lambda x: houseprice[x].value_counts().idxmax()).astype('int')
time_columns_summary['freq'] = time_columns_summary.index.map(lambda x: houseprice[x].value_counts().max()).astype('int')
time_columns_summary
print(f"{len(time_columns_summary)} Time columns with\n\nMissing Records   : {time_columns_summary[time_columns_summary['count'] < len(houseprice)].index.tolist()}\n")
for col in ['YearBuilt', 'YearRemodAdd', 'YrSold', 'GarageYrBlt']:
    print(f"{col} Summary:\n---------------------------\nNumber for Unique Values  : {time_columns_summary.loc[col, 'unique']}\nValue with most occurances: {time_columns_summary.loc[col, 'top']} ({time_columns_summary.loc[col, 'freq']})\nRange                     : {time_columns_summary.loc[col, 'min']} - {time_columns_summary.loc[col, 'max']} ({time_columns_summary.loc[col, 'max'] - time_columns_summary.loc[col, 'min']})\n\n")
cat_summary = houseprice.describe(include=[object]).transpose()
cat_summary['top'] = cat_summary.apply(lambda x: house_mapping[x.name][x['top']] if x.name in house_mapping and x['top'] in house_mapping[x.name] else x['top'], axis=1)
cat_summary
print(f"{len(cat_summary)} Categorical columns with\n\nMissing Records   : {cat_summary[cat_summary['count'] < len(houseprice)].index.tolist()}\n")
for col in ['MSZoning', 'SaleCondition', 'SaleCondition', 'Electrical', 'Heating', 'FireplaceQu']:
    print(f"{col} Summary:\n---------------------------\nNumber for Unique Values  : {cat_summary.loc[col, 'unique']}\nValue with most occurances: {cat_summary.loc[col, 'top']} ({cat_summary.loc[col, 'freq']})\n\n")
num_missing = houseprice.isna().sum()
num_missing = num_missing[num_missing > 0]
percent_missing = num_missing * 100 / houseprice.shape[0]
pd.concat([num_missing, percent_missing], axis=1, keys=['Missing Values', 'Percentage']).sort_values(by='Missing Values', ascending=False).T.style.background_gradient(axis=1)
num_missing = houseprice_test.isna().sum()
num_missing = num_missing[num_missing > 0]
percent_missing = num_missing * 100 / houseprice_test.shape[0]
pd.concat([num_missing, percent_missing], axis=1, keys=['Missing Values', 'Percentage']).sort_values(by='Missing Values', ascending=False).T.style.background_gradient(axis=1)
houseprice.groupby(['PoolQC'], as_index=False, dropna=False)['PoolArea'].min().T
houseprice['PoolQC'].fillna('NA', inplace=True)
houseprice_test['PoolQC'].fillna('NA', inplace=True)
houseprice['MiscFeature'].value_counts(dropna=False).to_frame().T
houseprice['MiscFeature'].fillna('NA', inplace=True)
houseprice_test['MiscFeature'].fillna('NA', inplace=True)
houseprice['Alley'].fillna('NA', inplace=True)
houseprice_test['Alley'].fillna('NA', inplace=True)
houseprice['Alley'].value_counts(dropna=False).to_frame().T
houseprice['Fence'].fillna('NA', inplace=True)
houseprice_test['Fence'].fillna('NA', inplace=True)
houseprice['Fence'].value_counts(dropna=False).to_frame().T
houseprice.groupby(['FireplaceQu'], as_index=False, dropna=False)['Fireplaces'].min().T
houseprice['FireplaceQu'].fillna('NA', inplace=True)
houseprice_test['FireplaceQu'].fillna('NA', inplace=True)
houseprice['LotFrontage'].fillna(0, inplace=True)
houseprice_test['LotFrontage'].fillna(0, inplace=True)
garage_columns = [col for col in houseprice.columns if col.startswith('Garage')]
houseprice[houseprice[garage_columns].isna().any(axis=1)][garage_columns].drop_duplicates()
for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    houseprice[col].fillna('NA', inplace=True)
    houseprice_test[col].fillna('NA', inplace=True)
houseprice['GarageYrBlt'].fillna(0, inplace=True)
houseprice_test['GarageYrBlt'].fillna(0, inplace=True)
bsmt_columns = [col for col in houseprice.columns if col.startswith('Bsmt')]
houseprice[houseprice[bsmt_columns].isna().any(axis=1)][bsmt_columns].drop_duplicates()
houseprice.loc[~pd.isnull(houseprice['BsmtCond']) & pd.isnull(houseprice['BsmtFinType2']), 'BsmtFinType2'] = 'Unf'
houseprice.loc[~pd.isnull(houseprice['BsmtCond']) & pd.isnull(houseprice['BsmtExposure']), 'BsmtExposure'] = 'No'
for col in ['BsmtExposure', 'BsmtFinType2', 'BsmtFinType1', 'BsmtCond', 'BsmtQual']:
    houseprice[col].fillna('NA', inplace=True)
    houseprice_test[col].fillna('NA', inplace=True)
houseprice['MasVnrArea'].fillna(0, inplace=True)
houseprice_test['MasVnrArea'].fillna(0, inplace=True)
houseprice['MasVnrType'].fillna('None', inplace=True)
houseprice_test['MasVnrType'].fillna('None', inplace=True)
print(' Record is null when \n ', houseprice.loc[houseprice['Electrical'].isna(), ['YearBuilt', 'YearRemodAdd']])
print('\n Getting the Electrical used during the period: \n', houseprice.loc[(houseprice['YearBuilt'] == 2006) | (houseprice['YearRemodAdd'] == 2007), 'Electrical'].value_counts())
houseprice['Electrical'].fillna('SBrkr', inplace=True)
houseprice_test['Electrical'].fillna('SBrkr', inplace=True)
num_missing = houseprice_test.isna().sum()
num_missing = num_missing[num_missing > 0]
percent_missing = num_missing * 100 / houseprice_test.shape[0]
pd.concat([num_missing, percent_missing], axis=1, keys=['Missing Values', 'Percentage']).sort_values(by='Missing Values', ascending=False).T.style.background_gradient(axis=1)
houseprice_test.loc[houseprice_test.isna().any(axis=1), num_missing.index.tolist()]
houseprice_test['MSZoning'].fillna('RL', inplace=True)
houseprice_test['Utilities'].fillna('AllPub', inplace=True)
houseprice_test['Exterior1st'].fillna('VinylSd', inplace=True)
houseprice_test['BsmtFinSF2'].fillna(0.0, inplace=True)
houseprice_test['BsmtUnfSF'].fillna(0.0, inplace=True)
houseprice_test['TotalBsmtSF'].fillna(0.0, inplace=True)
houseprice_test['BsmtFullBath'].fillna(0.0, inplace=True)
houseprice_test['BsmtHalfBath'].fillna(0.0, inplace=True)
houseprice_test['KitchenQual'].fillna('TA', inplace=True)
houseprice_test['Functional'].fillna('Typ', inplace=True)
houseprice_test['SaleType'].fillna('WD', inplace=True)
houseprice_test['GarageCars'].fillna(0.0, inplace=True)
houseprice_test['GarageArea'].fillna(0.0, inplace=True)

def plot_hist_scatter(df, feature, target='SalePrice', outlier=True, log=False, **kwargs):
    if log:
        feature_data = np.log1p(df[feature])
    else:
        feature_data = df[feature]
    plt.figure(figsize=(2 * 5.2, 1 * 3.2))
    print(f'skew={feature_data.skew()}, kurtosis={feature_data.kurtosis()}')
    plt.subplot(1, 2, 1)
    feature_data.hist(bins=50)
    if outlier:
        (cuttoff_min, cuttoff_max) = data_range_nooutlier(feature_data, log=False, **kwargs)
        print(f'Within range: between {cuttoff_min:.2f} and {cuttoff_max:.2f}')
        plt.axvspan(xmin=cuttoff_max, xmax=feature_data.max(), color='r', alpha=0.5)
    plt.xlabel(f"{('log' if log else '')}{feature}")
    plt.ylabel('count')
    plt.title(f"{('log' if log else '')}{feature} histogram")
    if feature != target:
        plt.subplot(1, 2, 2)
        plt.scatter(x=feature_data, y=df[target], color='orange', edgecolors='#000000', linewidths=0.5)
        plt.xlabel(feature)
        plt.ylabel(f'{target}')
        if outlier:
            plt.axvspan(xmin=cuttoff_max, xmax=feature_data.max(), color='r', alpha=0.5)


def plot_bar(df, columns):
    cols = 3
    rows = len(columns) // 3 + 1
    plt.figure(figsize=(cols * 6.7, rows * 3.75))
    i = 0
    for row in range(rows):
        for col in range(cols):
            index = cols * row + col
            if index >= len(columns):
                break
            plt.subplot(rows, cols, index + 1)
            df.groupby(columns[i]).size().plot(kind='bar')
            i += 1

def plot_box(df, y, columns):
    cols = 3
    rows = len(columns) // 3 + 1
    plt.figure(figsize=(cols * 5.5, rows * 3.5))
    i = 0
    for row in range(rows):
        for col in range(cols):
            index = cols * row + col
            if index >= len(columns):
                break
            plt.subplot(rows, cols, index + 1)
            sns.boxplot(x=columns[i], y=y, data=df)
            i += 1

def plot_hist(df, columns):
    cols = 3
    rows = len(columns) // 3 + 1
    plt.figure(figsize=(cols * 5.5, rows * 3.5))
    i = 0
    for row in range(rows):
        for col in range(cols):
            index = cols * row + col
            if index >= len(columns):
                break
            plt.subplot(rows, cols, index + 1)
            df[columns[i]].hist(bins=50)
            plt.ylabel(columns[i])
            i += 1

def plot_scatter(df, y, columns):
    cols = 3
    rows = len(columns) // 3 + 1
    plt.figure(figsize=(cols * 7.5, rows * 5.2))
    i = 0
    for row in range(rows):
        for col in range(cols):
            index = cols * row + col
            if index >= len(columns):
                break
            plt.subplot(rows, cols, index + 1)
            sns.scatterplot(x=columns[i], y=y, data=df)
            i += 1

def calculateAnova(inpData, y, catCols, target):
    inpData = inpData.join(y)
    from scipy.stats import f_oneway
    CatColumnList = []
    for cat in catCols:
        CatGroupList = inpData.groupby(cat)[target].apply(list)
        anova = f_oneway(*CatGroupList)
        if anova[1] < 0.05:
            print('The column ', cat, ' is correlated with ', target, ' | P-Value: ', anova[1])
            CatColumnList.append(cat)
        else:
            print('The column ', cat, ' is NOT correlated with ', target, ' | P-Value: ', anova[1])
    return CatColumnList
cuttoff = dict()
for col in numerical_columns:
    cuttoff[col] = dict()
    (cuttoff[col]['min'], cuttoff[col]['max']) = (houseprice[col].min(), houseprice[col].max())
    (cuttoff[col]['cutoffmin'], cuttoff[col]['cutoffmax']) = data_range_nooutlier(houseprice[col], continuous=True)
    (cuttoff[col]['p25'], cuttoff[col]['p75']) = np.percentile(houseprice[col], [25, 75])
    cuttoff[col]['cutoffmin'] = cuttoff[col]['cutoffmin'] if cuttoff[col]['min'] < cuttoff[col]['cutoffmin'] else cuttoff[col]['min']
    plot_hist_scatter(houseprice, col, continuous=True)
pd.DataFrame.from_dict(cuttoff).T[['min', 'cutoffmin', 'p25', 'p75', 'cutoffmax', 'max']]
cor_df = houseprice[numerical_columns + ['SalePrice']].corr()['SalePrice'].sort_values(key=abs, ascending=False).to_frame()
cor_df.style.background_gradient(axis=1)
threshold_ratio = 1.2
for col in cor_df[abs(cor_df['SalePrice']) >= 0.5].index.tolist():
    if col != 'SalePrice':
        print(f"Adding cuttoff max for {col} : {cuttoff[col]['cutoffmax'] * threshold_ratio:.2f}")
        houseprice = houseprice[houseprice[col] <= cuttoff[col]['cutoffmax'] * threshold_ratio]
print('Shape after removing outlier ', houseprice.shape)
X = houseprice.copy()
X_test = houseprice_test.copy()
y = X.pop('SalePrice')
(fig, axes) = plt.subplots(1, 2, figsize=(15, 3), sharey=False)
fig.suptitle('Histogram')
sns.histplot(data=y, kde=True, ax=axes[0])
axes[0].set_title('SalePrice ')
sns.histplot(ax=axes[1], data=np.log(y), kde=True)
axes[1].set_title('Log of SalePrice')
axes[1].set_xlabel('Log SalePrice')

SalePriceSF = y / X['GrLivArea']
plt.hist(SalePriceSF, bins=15, color='gold')
plt.title('Sale Price per Square Foot')
plt.ylabel('Number of Sales')
plt.xlabel('Price per square feet')
print('Average Sale Price per square feet: $', SalePriceSF.mean())
cat_columns = cat_summary.index.tolist()
plot_bar(X, numerical_discrete + time_columns + cat_columns)
plot_hist(X, numerical_columns)
X.columns
plot_box(X, y, numerical_discrete + time_columns + cat_columns)
(fig, ax) = plt.subplots(figsize=(20, 7))
sns.heatmap(X[numerical_columns].corr(), cmap='coolwarm', annot=True, annot_kws={'size': 10})

correlations = X.corr()
attrs = correlations.iloc[:-1, :-1]
threshold = 0.5
important_corrs = attrs[abs(attrs) > threshold][attrs != 1.0].unstack().dropna().to_dict()
unique_important_corrs = pd.DataFrame(list(set([(tuple(sorted(key)), important_corrs[key]) for key in important_corrs])), columns=['Attribute Pair', 'Correlation'])
unique_important_corrs = unique_important_corrs.iloc[abs(unique_important_corrs['Correlation']).argsort()[::-1]]
unique_important_corrs.style.background_gradient(axis=0)
sns.scatterplot(y=y, x='GarageArea', data=X)

sns.regplot(x='OverallQual', y=y, data=X, robust=True)
sns.regplot(y=y, x='YearBuilt', data=X)
sns.regplot(x='TotalBsmtSF', y=y, data=X)
sns.regplot(y=y, x='GarageYrBlt', data=X)
X_test[X_test.isna().any(axis=1)]
X['LivLotRatio'] = X.GrLivArea / (X.LotArea + 1)
X_test['LivLotRatio'] = X_test.GrLivArea / (X.LotArea + 1)
X['Spaciousness'] = (X['1stFlrSF'] + X['2ndFlrSF']) / (X.TotRmsAbvGrd + 1)
X_test['Spaciousness'] = (X_test['1stFlrSF'] + X_test['2ndFlrSF']) / (X_test.TotRmsAbvGrd + 1)
X['MedNhbdArea'] = X.groupby('Neighborhood')['GrLivArea'].transform('median')
X_test['MedNhbdArea'] = X_test.groupby('Neighborhood')['GrLivArea'].transform('median')
X['PorchTypes'] = X[['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']].gt(0.0).sum(axis=1)
X_test['PorchTypes'] = X_test[['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']].gt(0.0).sum(axis=1)
numerical_columns = numerical_columns + ['LivLotRatio', 'Spaciousness', 'MedNhbdArea']
plot_scatter(X, y, numerical_columns)
corr_mat = X[numerical_columns].join(y).corr()
selected_numerical_columns = corr_mat['SalePrice'][abs(corr_mat['SalePrice']) >= 0.5].index.tolist()
nonselected_numerical_columns = corr_mat['SalePrice'][abs(corr_mat['SalePrice']) < 0.5].index.tolist()
selected_numerical_columns
selected_categorical_cols = calculateAnova(X, y, numerical_discrete + nonselected_numerical_columns + time_columns + cat_columns, 'SalePrice')
selected_categorical_cols
selected_col = [col for col in X.columns if col in selected_numerical_columns + selected_categorical_cols]
X = X[selected_col].copy()
X_test = X_test[selected_col].copy()
correlations = X.corr()
attrs = correlations.iloc[:-1, :-1]
threshold = 0.5
important_corrs = attrs[abs(attrs) > threshold][attrs != 1.0].unstack().dropna().to_dict()
unique_important_corrs = pd.DataFrame(list(set([(tuple(sorted(key)), important_corrs[key]) for key in important_corrs])), columns=['Attribute Pair', 'Correlation'])
unique_important_corrs = unique_important_corrs.iloc[abs(unique_important_corrs['Correlation']).argsort()[::-1]]
unique_important_corrs.style.background_gradient(axis=0)
X.drop(['GarageCars', 'GarageYrBlt', 'TotRmsAbvGrd'], axis=1, inplace=True)
X_test.drop(['GarageCars', 'GarageYrBlt', 'TotRmsAbvGrd'], axis=1, inplace=True)

def make_mi_scores(X, y):
    """Estimate mutual information for a continuous target variable."""
    X = X.copy()
    for colname in X.select_dtypes(['object', 'category']):
        (X[colname], _) = X[colname].factorize()
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name='MI Scores', index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores
mi_scores = make_mi_scores(X, y)
mi_scores.head()

def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.figure(figsize=(10, 15))
    clrs = ['grey' if x < 0.01 else 'blue' for x in scores]
    plt.barh(width, scores, color=clrs)
    plt.yticks(width, ticks)
    plt.title('Mutual Information Scores')
plot_mi_scores(mi_scores)

def corrplot(df, method='pearson', annot=False, **kwargs):
    plt.figure(figsize=(25, 40))
    sns.clustermap(df.corr(method), vmin=-1.0, vmax=1.0, cmap='icefire', method='complete', annot=annot, **kwargs)
corrplot(X.join(y))
from sklearn.preprocessing import StandardScaler

def transform_dataset(dataset, test_data):

    def transform_categoricals(dataset):
        mp = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0}
        dataset['ExterQual'] = dataset['ExterQual'].map(mp)
        dataset['HeatingQC'] = dataset['HeatingQC'].map(mp)
        dataset['KitchenQual'] = dataset['KitchenQual'].map(mp)
        mp = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
        dataset['BsmtQual'] = dataset['BsmtQual'].map(mp)
        dataset['BsmtCond'] = dataset['BsmtCond'].map(mp)
        dataset['BsmtExposure'] = dataset['BsmtExposure'].map({'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0})
        mp = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0}
        dataset['BsmtFinType1'] = dataset['BsmtFinType1'].map(mp)
        dataset['BsmtFinType2'] = dataset['BsmtFinType2'].map(mp)
        dataset['CentralAir'] = dataset['CentralAir'].map({'Y': 1, 'N': 0})
        dataset['Functional'] = dataset['Functional'].map({'Typ': 7, 'Min1': 6, 'Min2': 5, 'Mod': 4, 'Maj1': 3, 'Maj2': 2, 'Sev': 1, 'Sal': 0})
        dataset['FireplaceQu'] = dataset['FireplaceQu'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
        dataset['GarageFinish'] = dataset['GarageFinish'].map({'Fin': 3, 'RFn': 2, 'Unf': 1, 'NA': 0})
        dataset['GarageQual'] = dataset['GarageQual'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
        dataset['GarageCond'] = dataset['GarageCond'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
        dataset['Fence'] = dataset['Fence'].map({'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'NA': 0})
        return dataset
    numerical_columns = dataset.select_dtypes(include=[np.number]).columns.tolist()
    dataset = transform_categoricals(dataset)
    test_data = transform_categoricals(test_data)
    scaler = StandardScaler()
    dataset.loc[:, numerical_columns] = scaler.fit_transform(dataset.loc[:, numerical_columns])
    test_data.loc[:, numerical_columns] = scaler.transform(test_data.loc[:, numerical_columns])
    cat_cols = dataset.select_dtypes(exclude=[np.number]).columns
    dataset = dataset.loc[:, mi_scores >= 0.05]
    test_data = test_data.loc[:, mi_scores >= 0.05]
    dataset = pd.get_dummies(dataset)
    test_data = pd.get_dummies(test_data)
    return (dataset, test_data)
(X, X_test) = transform_dataset(X, X_test)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
(Xtrain, Xtest, ytrain, ytest) = train_test_split(X, y, test_size=0.1, random_state=3)
parameter_space = {'alpha': [1, 10, 100, 290, 500], 'fit_intercept': [True, False], 'solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}
clf = GridSearchCV(Ridge(random_state=3), parameter_space, n_jobs=-1, cv=5, scoring='neg_mean_squared_error')