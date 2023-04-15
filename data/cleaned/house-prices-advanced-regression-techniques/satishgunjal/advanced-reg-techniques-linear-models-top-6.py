import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from scipy.special import boxcox1p
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))
rcParams['figure.figsize'] = (12, 6)
print('Input data files,')
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
print(f'Shape of traning data= {train.shape}')
train.head()
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print(f'Shape of test data= {test.shape}')
test.head()
train.drop('Id', inplace=True, axis=1)
test.drop('Id', inplace=True, axis=1)
print(f'After dropping Id feature, shape of train data: {train.shape}, test data: {test.shape}')
train_corr = train.corr(method='pearson')
(f, ax) = plt.subplots(figsize=(25, 25))
mask = np.triu(np.ones_like(train_corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
ax = sns.heatmap(train_corr, vmin=-1, vmax=1, mask=mask, cmap=cmap, center=0, annot=True, square=True, linewidths=0.5, cbar_kws={'shrink': 0.5, 'orientation': 'vertical'})
(figure, ax) = plt.subplots(1, 2, figsize=(24, 10))
sns.boxplot(data=train, x='OverallQual', y='SalePrice', ax=ax[0])
sns.violinplot(data=train, x='OverallQual', y='SalePrice', ax=ax[1])

(figure, ax) = plt.subplots(2, 1, figsize=(24, 10))
figure.tight_layout(pad=5.0)
sns.barplot(ax=ax[0], x='YearBuilt', y='SalePrice', data=train)
ax[0].set(xlabel='YearBuilt', ylabel='SalePrice')
ax[0].set_title('YearBuilt vs SalePrice (Correlation coefficient: 0.52)')
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=90)
sns.barplot(ax=ax[1], x='YearRemodAdd', y='SalePrice', data=train)
ax[1].set(xlabel='YearRemodAdd', ylabel='SalePrice')
ax[1].set_title('YearRemodAdd vs SalePrice (Correlation coefficient: 0.51)')
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=90)

(figure, ax) = plt.subplots(2, 2, figsize=(24, 8))
figure.tight_layout(pad=4.0)
sns.regplot(data=train, x='TotalBsmtSF', y='SalePrice', scatter_kws={'alpha': 0.2}, line_kws={'color': 'blue'}, ax=ax[0, 0])
ax[0, 0].set_title('TotalBsmtSF vs SalePrice (Correlation coefficient: 0.61)', fontsize=12)
sns.regplot(data=train, x='1stFlrSF', y='SalePrice', scatter_kws={'alpha': 0.2}, line_kws={'color': 'blue'}, ax=ax[0, 1])
ax[0, 1].set_title('1stFlrSF vs SalePrice (Correlation coefficient: 0.61)', fontsize=12)
sns.regplot(data=train, x='GrLivArea', y='SalePrice', scatter_kws={'alpha': 0.2}, line_kws={'color': 'blue'}, ax=ax[1, 0])
ax[1, 0].set_title('\nGrLivArea vs SalePrice (Correlation coefficient: 0.71)', fontsize=12)
sns.regplot(data=train, x='GarageArea', y='SalePrice', scatter_kws={'alpha': 0.2}, line_kws={'color': 'blue'}, ax=ax[1, 1])
ax[1, 1].set_title('\nGarageArea vs SalePrice (Correlation coefficient: 0.62)', fontsize=12)

(figure, ax) = plt.subplots(3, 2, figsize=(24, 12))
figure.tight_layout(pad=4.0)
sns.boxplot(data=train, x='FullBath', y='SalePrice', ax=ax[0, 0])
sns.violinplot(data=train, x='FullBath', y='SalePrice', ax=ax[0, 1])
ax[0, 0].set_title('FullBath vs SalePrice (Correlation coefficient: 0.56)', fontsize=12)
ax[0, 1].set_title('FullBath vs SalePrice (Correlation coefficient: 0.56)', fontsize=12)
sns.boxplot(data=train, x='TotRmsAbvGrd', y='SalePrice', ax=ax[1, 0])
sns.violinplot(data=train, x='TotRmsAbvGrd', y='SalePrice', ax=ax[1, 1])
ax[1, 0].set_title('TotRmsAbvGrd vs SalePrice (Correlation coefficient: 0.53)', fontsize=12)
ax[1, 1].set_title('TotRmsAbvGrd vs SalePrice (Correlation coefficient: 0.53)', fontsize=12)
sns.boxplot(data=train, x='GarageCars', y='SalePrice', ax=ax[2, 0])
sns.violinplot(data=train, x='GarageCars', y='SalePrice', ax=ax[2, 1])
ax[2, 0].set_title('GarageCars vs SalePrice (Correlation coefficient: 0.64)', fontsize=12)
ax[2, 1].set_title('GarageCars vs SalePrice (Correlation coefficient: 0.64)', fontsize=12)

min_percentile = 0.001
max_percentile = 0.999
features = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
target = 'SalePrice'
nrows = int(np.ceil(len(features) / 2))
ncols = 2

def detect_and_remove_outliers(inline_delete=True):
    global train
    (fig, ax) = plt.subplots(nrows=nrows, ncols=ncols, figsize=(24, nrows * 6))
    outliers = []
    cnt = 0
    for row in range(0, nrows):
        for col in range(0, ncols):
            (min_thresold, max_thresold) = train[features[cnt]].quantile([min_percentile, max_percentile])
            df_outliers = train[(train[features[cnt]] < min_thresold) | (train[features[cnt]] > max_thresold)]
            outliers = outliers + df_outliers.index.tolist()
            ax[row][col].scatter(x=train[features[cnt]], y=train[target])
            ax[row][col].scatter(x=df_outliers[features[cnt]], y=df_outliers[target], marker='o', edgecolor='red', s=100)
            ax[row][col].set_xlabel(features[cnt])
            ax[row][col].set_ylabel(target)
            ax[row][col].set_title('Outlier detection for feature ' + features[cnt])
            if inline_delete:
                train = train.drop(df_outliers.index.tolist())
                train.reset_index(drop=True, inplace=True)
            cnt = cnt + 1
            if cnt >= len(features):
                break

    print(f'outliers: {outliers}')
    unique_outliers = list(set(outliers))
    print(f'unique_outliers: {unique_outliers}')
    if inline_delete == False:
        print(f'Shape of train data= {train.shape}')
        train = train.drop(unique_outliers)
        train.reset_index(drop=True, inplace=True)
        print(f'Shape of train data= {train.shape}')
detect_and_remove_outliers(inline_delete=False)
(fig, ax) = plt.subplots(nrows=nrows, ncols=ncols, figsize=(24, nrows * 6))
outliers = []
cnt = 0
for row in range(0, nrows):
    for col in range(0, ncols):
        sns.regplot(data=train, x=features[cnt], y=target, scatter_kws={'alpha': 0.2}, line_kws={'color': 'blue'}, ax=ax[row, col])
        ax[row, col].set_title("Regplot after removing outlier's from feature " + features[cnt], fontsize=12)
        cnt = cnt + 1
        if cnt >= len(features):
            break

y_train = train.SalePrice
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print('Shape of all_data= {}'.format(all_data.shape))
all_data.head()
all_data = all_data.drop(['Utilities', 'Street', 'PoolQC'], axis=1)
print('Shape of all_data= {}'.format(all_data.shape))
for col in ('MSSubClass', 'YrSold', 'MoSold'):
    all_data[col] = all_data[col].astype(str)
col_na = all_data.columns[all_data.isnull().any()]
all_data_na_cnt = all_data[col_na].isnull().sum()
all_data_na = all_data[col_na].isnull().sum() / len(all_data) * 100
all_data_na = pd.DataFrame({'Total Null Val': all_data_na_cnt, 'Null Value %': all_data_na})
all_data_na = all_data_na.sort_values(by='Null Value %', ascending=False)
all_data_na
sns.barplot(x=all_data_na.index, y='Null Value %', data=all_data_na)
plt.xticks(rotation=90)
plt.title('Column name with null value %')

sns.heatmap(all_data[col_na].isnull())

for col in ('MiscFeature', 'Alley', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'Fence', 'GarageType', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'MSSubClass'):
    all_data[col] = all_data[col].fillna('None')
    print(f'Feature: {col}, Null Count: {all_data[col].isnull().sum()}, Unique Values: {all_data[col].unique()}')
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'):
    all_data[col] = all_data[col].fillna(0)
all_data['Functional'] = all_data['Functional'].fillna('Typ')
for col in ('MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType'):
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
    print(f'Feature: {col}, Null Count: {all_data[col].isnull().sum()}, Unique Values: {all_data[col].unique()}')
print(f'Shape of data: {all_data.shape}')
print(f'Count of null values: {all_data.isnull().sum().sum()}')
sns.heatmap(all_data.isnull())
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def get_highest_vif_feature(df, thresh=5):
    """
    Ref: https://stackoverflow.com/questions/42658379/variance-inflation-factor-in-python
    
    Calculates VIF each feature in a pandas dataframe
    A constant must be added to variance_inflation_factor or the results will be incorrect

    :param df: the pandas dataframe containing only the predictor features, not the response variable
    :param thresh: the max VIF value before the feature is removed from the dataframe
    :return: dataframe with features removed
    """
    const = add_constant(df)
    print(f'Shape of data after adding const column: {const.shape}')
    cols = const.columns
    vif_df = pd.Series([variance_inflation_factor(const.values, i) for i in range(const.shape[1])], index=const.columns).to_frame()
    vif_df = vif_df.sort_values(by=0, ascending=False).rename(columns={0: 'VIF'})
    vif_df = vif_df.drop('const')
    vif_df = vif_df[vif_df['VIF'] > thresh]
    if vif_df.empty:
        print('DataFrame is empty!')
        return None
    else:
        print(f'\nFeatures above VIF threshold: {vif_df.to_dict()}')
        return list(vif_df.index)[0]
        print(f'Lets delete the feature with highest VIF value: {list(vif_df.index)[0]}')
print(f'Shape of input data: {all_data.shape}')
numeric_feats = all_data.dtypes[all_data.dtypes != 'object'].index
print(f'Calculating VIF for {len(numeric_feats)} numerical features')
df_numeric = all_data[numeric_feats]
print(f'Shape of df_numeric: {df_numeric.shape}')
feature_to_drop = None
feature_to_drop_list = []
while True:
    feature_to_drop = get_highest_vif_feature(df_numeric, thresh=5)
    print(f'feature_to_drop: {feature_to_drop}')
    if feature_to_drop is None:
        print('No more features to drop!')
        break
    else:
        feature_to_drop_list.append(feature_to_drop)
        df_numeric = df_numeric.drop(feature_to_drop, axis=1)
        print(f'Feature {feature_to_drop} droped from df_numeric')
print(f'\nfeature_to_drop_list: {feature_to_drop_list}')
print(f'Shape of traning data= {all_data.shape}')
all_data = all_data.drop(['LowQualFinSF'], axis=1)
all_data.reset_index(drop=True, inplace=True)
print(f'Shape of traning data= {all_data.shape}')
cat_feats = all_data.dtypes[all_data.dtypes == 'object'].index
numeric_feats = all_data.dtypes[all_data.dtypes != 'object'].index
print(f'Number of categorical features: {len(cat_feats)}, Numerical features: {len(numeric_feats)}')
skew_features = all_data[numeric_feats].apply(lambda x: stats.skew(x)).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew': skew_features})
print(f'Skew in numerical features. Shape of skewness: {skewness.shape}')
skewness.head(10)
high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index
for i in skew_index:
    all_data[i] = boxcox1p(all_data[i], stats.boxcox_normmax(all_data[i] + 1))
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['TotalSF1'] = all_data['BsmtFinSF1'] + all_data['BsmtFinSF2'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['YrBltAndRemod'] = all_data['YearBuilt'] + all_data['YearRemodAdd']
all_data['TotalBathrooms'] = all_data['FullBath'] + 0.5 * all_data['HalfBath'] + all_data['BsmtFullBath'] + 0.5 * all_data['BsmtHalfBath']
all_data['TotalPorchSF'] = all_data['OpenPorchSF'] + all_data['3SsnPorch'] + all_data['EnclosedPorch'] + all_data['ScreenPorch'] + all_data['WoodDeckSF']
print(f'Shape all_data: {all_data.shape}')
all_data['haspool'] = all_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['has2ndfloor'] = all_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['hasgarage'] = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['hasbsmt'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['hasfireplace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
print(f'Shape all_data: {all_data.shape}')
cat_feats = all_data.dtypes[all_data.dtypes == 'object'].index
numeric_feats = all_data.dtypes[all_data.dtypes != 'object'].index
print(f'Number of categorical features: {len(cat_feats)}, Numerical features: {len(numeric_feats)}')
print(f'\nList of cetagorical features: {cat_feats.to_list()}\n\nList of numerical features: {numeric_feats.to_list()}')
cat_feats_nominal = ['MSSubClass', 'MSZoning', 'Neighborhood', 'Condition1', 'Condition2', 'HouseStyle', 'CentralAir', 'MiscFeature', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition', 'Electrical', 'MasVnrType', 'Exterior1st', 'Exterior2nd', 'Heating', 'Foundation']
cat_feats_ordinal = ['Alley', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope', 'BldgType', 'RoofStyle', 'RoofMatl', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType2', 'BsmtFinType1', 'HeatingQC', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'Fence']
numeric_feats_cont = ['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageYrBlt', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'TotalSF', 'TotalSF1', 'YrBltAndRemod', 'TotalBathrooms', 'TotalPorchSF']
numeric_feats_ordinal = ['OverallQual', 'OverallCond']
numeric_feats_descrete = ['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'haspool', 'has2ndfloor', 'hasgarage', 'hasbsmt', 'hasfireplace']
print(f'Number of cat_feats_nominal: {len(cat_feats_nominal)}, cat_feats_ordinal: {len(cat_feats_ordinal)}, numeric_feats_cont: {len(numeric_feats_cont)}, numeric_feats_ordinal: {len(numeric_feats_ordinal)}, numeric_feats_descrete: {len(numeric_feats_descrete)} ')
print(f'List of categorical ordinal features: {cat_feats_ordinal}')
all_data['Alley'].replace(to_replace=['None', 'Grvl', 'Pave'], value=[0, 1, 2], inplace=True)
all_data['LotShape'].replace(to_replace=['Reg', 'IR1', 'IR2', 'IR3'], value=[3, 2, 1, 0], inplace=True)
all_data['LandContour'].replace(to_replace=['Lvl', 'Bnk', 'Low', 'HLS'], value=[3, 2, 1, 0], inplace=True)
all_data['LotConfig'].replace(to_replace=['Inside', 'FR2', 'Corner', 'CulDSac', 'FR3'], value=[0, 3, 1, 2, 4], inplace=True)
all_data['LandSlope'].replace(to_replace=['Gtl', 'Mod', 'Sev'], value=[2, 1, 0], inplace=True)
all_data['BldgType'].replace(to_replace=['1Fam', '2fmCon', 'Duplex', 'TwnhsE', 'Twnhs'], value=[4, 3, 2, 1, 0], inplace=True)
all_data['RoofStyle'].replace(to_replace=['Gable', 'Hip', 'Gambrel', 'Mansard', 'Flat', 'Shed'], value=[4, 2, 3, 1, 5, 0], inplace=True)
all_data['RoofMatl'].replace(to_replace=['ClyTile', 'CompShg', 'Membran', 'Metal', 'Roll', 'Tar&Grv', 'WdShake', 'WdShngl'], value=[7, 6, 5, 4, 3, 2, 1, 0], inplace=True)
all_data['ExterQual'].replace(to_replace=['Ex', 'Gd', 'TA', 'Fa'], value=[3, 2, 1, 0], inplace=True)
all_data['ExterCond'].replace(to_replace=['Ex', 'Gd', 'TA', 'Fa', 'Po'], value=[4, 3, 2, 1, 0], inplace=True)
all_data['BsmtQual'].replace(to_replace=['Ex', 'Gd', 'TA', 'Fa', 'None'], value=[4, 3, 2, 1, 0], inplace=True)
all_data['BsmtCond'].replace(to_replace=['Gd', 'TA', 'Fa', 'Po', 'None'], value=[4, 3, 2, 1, 0], inplace=True)
all_data['BsmtExposure'].replace(to_replace=['Gd', 'Av', 'Mn', 'No', 'None'], value=[4, 3, 2, 1, 0], inplace=True)
all_data['BsmtFinType1'].replace(to_replace=['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'None'], value=[6, 5, 4, 3, 2, 1, 0], inplace=True)
all_data['BsmtFinType2'].replace(to_replace=['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'None'], value=[6, 5, 4, 3, 2, 1, 0], inplace=True)
all_data['HeatingQC'].replace(to_replace=['Ex', 'Gd', 'TA', 'Fa', 'Po'], value=[4, 3, 2, 1, 0], inplace=True)
all_data['KitchenQual'].replace(to_replace=['Ex', 'Gd', 'TA', 'Fa'], value=[3, 2, 1, 0], inplace=True)
all_data['Functional'].replace(to_replace=['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev'], value=[6, 5, 4, 3, 2, 1, 0], inplace=True)
all_data['FireplaceQu'].replace(to_replace=['Ex', 'Gd', 'TA', 'Fa', 'Po', 'None'], value=[5, 4, 3, 2, 1, 0], inplace=True)
all_data['GarageType'].replace(to_replace=['2Types', 'Attchd', 'Basment', 'BuiltIn', 'CarPort', 'Detchd', 'None'], value=[6, 5, 4, 3, 2, 1, 0], inplace=True)
all_data['GarageFinish'].replace(to_replace=['Fin', 'RFn', 'Unf', 'None'], value=[3, 2, 1, 0], inplace=True)
all_data['GarageQual'].replace(to_replace=['Ex', 'Gd', 'TA', 'Fa', 'Po', 'None'], value=[5, 4, 3, 2, 1, 0], inplace=True)
all_data['GarageCond'].replace(to_replace=['Ex', 'Gd', 'TA', 'Fa', 'Po', 'None'], value=[5, 4, 3, 2, 1, 0], inplace=True)
all_data['PavedDrive'].replace(to_replace=['Y', 'P', 'N'], value=[2, 1, 0], inplace=True)
all_data['Fence'].replace(to_replace=['GdPrv', 'MnPrv', 'GdWo', 'MnWw', 'None'], value=[4, 3, 2, 1, 0], inplace=True)
print(f'\nShape of all_data: {all_data.shape}')
all_data.head()
print(f'List of categorical nominal features: {cat_feats_nominal}')
cat_feats_nominal_one_hot = pd.get_dummies(all_data[cat_feats_nominal], drop_first=True).reset_index(drop=True)
print(f'Shape of cat_feats_nominal_one_hot: {cat_feats_nominal_one_hot.shape}')
cat_feats_nominal_one_hot.head()
all_data = all_data.drop(cat_feats_nominal, axis='columns')
all_data = pd.concat([all_data, cat_feats_nominal_one_hot], axis='columns')
print(f'Shape of all_data: {all_data.shape}')
all_data.head()
train = all_data[:len(y_train)]
test = all_data[len(y_train):]
print(f'Shape of train: {train.shape}, test:{test.shape}')

def distplot_probplot():
    """
     Plot histogram using normal distribution and probability plot
    """
    (fig, ax) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('SalesPrice Normal Distribution and Probability Plot', fontsize=15)
    sns.distplot(y_train, fit=stats.norm, label='test_label2', ax=ax[0])
    stats.probplot(y_train, plot=ax[1])
