from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
np.random.seed(42)
plt.rcParams.update({'figure.max_open_warning': 0})
pd.options.mode.chained_assignment = None
pd.set_option('display.max_rows', 250)
pd.set_option('display.max_columns', 250)
from sklearn.model_selection import KFold, cross_val_score
from skopt.space import Real, Integer
from skopt import BayesSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
import lightgbm as lgb
import xgboost as xgb
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train_id = _input1.Id
test_id = _input0.Id
print('Train set: ', _input1.shape)
print('Test set: ', _input0.shape)
_input1.info()
_input1.head()
_input1.describe().round(3)
_input1.describe(include=['O'])
df = pd.concat([_input1, _input0])
df

def col_types(df):
    num_cols = df.loc[:, df.dtypes != 'object'].columns.tolist()
    cat_cols = df.loc[:, df.dtypes == 'object'].columns.tolist()
    ord_cols = []
    for col in num_cols:
        if df[col].value_counts().size < 20:
            ord_cols.append(col)
    num_cols = [x for x in num_cols if x not in ord_cols + ['Id', 'SalePrice']]
    return (num_cols, cat_cols, ord_cols)

def missing(df):
    miss = pd.DataFrame({'no_missing_values': df.isnull().sum(), 'missing_value_ratio': (df.isnull().sum() / df.shape[0]).round(4), 'missing_in_train': df[df.SalePrice.notnull()].isnull().sum(), 'missing_in_test': df[df.SalePrice.isnull()].isnull().sum()})
    return miss[miss.no_missing_values > 0].sort_values('no_missing_values', ascending=False)
missing(df)
df[df.GarageFinish.isnull() & df.GarageType.notnull()]
df.loc[df.GarageFinish.isnull() & df.GarageType.notnull(), 'GarageFinish'] = 'Fin'
df.loc[df.GarageCars.isnull() & df.GarageType.notnull(), 'GarageCars'] = 1
df.loc[df.GarageQual.isnull() & df.GarageType.notnull(), 'GarageQual'] = 'TA'
df.loc[df.GarageCond.isnull() & df.GarageType.notnull(), 'GarageCond'] = 'TA'
df[df.GarageYrBlt == df.YearBuilt].shape[0]
df.loc[df.GarageYrBlt.isnull() & df.GarageType.notnull(), 'GarageYrBlt'] = df.loc[df.GarageYrBlt.isnull() & df.GarageType.notnull()].YearBuilt
df[(df.GarageType == 'Detchd') & (df.YearBuilt < 1930) & (df.YearRemodAdd < 2000) & (df.YearRemodAdd > 1980) & (df.GarageCars == 1)].GarageArea.median()
df.loc[df.GarageArea.isnull() & df.GarageType.notnull(), 'GarageArea'] = 234
(num_cols, cat_cols, ord_cols) = col_types(df)
none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType', 'BsmtExposure', 'BsmtCond', 'BsmtQual', 'BsmtFinType1', 'MasVnrType', 'BsmtFinType2']
for col in none_cols:
    df[col] = df[col].fillna('None', inplace=False)
missing(df)
df.loc[df.MasVnrArea.isnull() & (df.MasVnrType == 'None'), 'MasVnrArea'] = 0
for col in ['BsmtFullBath', 'BsmtHalfBath', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'GarageYrBlt']:
    df[col] = df[col].fillna(0, inplace=False)
df['TotalBsmtSF'] = df['TotalBsmtSF'].fillna(df['BsmtFinSF1'] + df['BsmtFinSF2'] + df['BsmtUnfSF'], inplace=False)
missing(df)
print(df.MSZoning.value_counts())
df[df.MSZoning.isnull()]
df.groupby('Neighborhood').MSZoning.value_counts()
df.loc[df.MSZoning.isnull() & (df.Neighborhood == 'IDOTRR'), 'MSZoning'] = 'C (all)'
df.loc[df.MSZoning.isnull() & (df.Neighborhood == 'Mitchel'), 'MSZoning'] = 'RL'
df.KitchenQual.value_counts()
print(df.groupby(['OverallQual', 'KitchenAbvGr']).KitchenQual.value_counts())
df.loc[df.KitchenQual.isnull() & (df.OverallQual == 5) & (df.KitchenAbvGr == 1), 'KitchenQual'] = 'TA'
print(df.SaleType.value_counts())
df[df.SaleType.isnull()]
print(df.groupby(['Neighborhood', 'SaleCondition']).SaleType.value_counts())
df.loc[df.SaleType.isnull() & (df.Neighborhood == 'Sawyer') & (df.SaleCondition == 'Normal'), 'SaleType'] = 'WD'
print(df.Electrical.value_counts())
df[df.Electrical.isnull()]
df[df.YearBuilt > 2005].Electrical.value_counts()
df.Electrical = df.Electrical.fillna('SBrkr', inplace=False)
print(df.Exterior1st.value_counts())
df[df.Exterior1st.isnull()]
print(df[df.RoofMatl == 'Tar&Grv'].Exterior1st.value_counts())
df.Exterior1st = df.Exterior1st.fillna('Plywood', inplace=False)
print(df[df.RoofMatl == 'Tar&Grv'].Exterior2nd.value_counts())
df.Exterior2nd = df.Exterior2nd.fillna('Plywood', inplace=False)
print(df.Functional.value_counts())
df[df.Functional.isnull()]
df[(df.Neighborhood == 'IDOTRR') & (df.OverallQual < 5) & (df.YearRemodAdd < 1960) & (df.ExterQual == 'Fa')].Functional.value_counts()
df.Functional = df.Functional.fillna('Mod', inplace=False)
df['LotFrontage'] = df['LotFrontage'].fillna(df.groupby(['Neighborhood', 'LotShape', 'LotConfig'])['LotFrontage'].transform('median'))
df['LotFrontage'] = df['LotFrontage'].fillna(df.groupby(['Neighborhood', 'LotShape'])['LotFrontage'].transform('median'))
df['LotFrontage'] = df['LotFrontage'].fillna(df.groupby('Neighborhood')['LotFrontage'].transform('median'))
df.Utilities = df.Utilities.fillna(df.Utilities.mode()[0], inplace=False)
missing(df)
df['MSSubClass'] = df['MSSubClass'].astype('str')
df.loc[df.GarageYrBlt == 2207, 'GarageYrBlt'] = 2007
df.loc[df.Exterior2nd == 'CmentBd', 'Exterior2nd'] = 'CemntBd'
df.loc[df.Exterior2nd == 'Wd Shng', 'Exterior2nd'] = 'WdShing'
df.loc[df.Exterior2nd == 'Brk Cmn', 'Exterior2nd'] = 'BrkComm'
dff = df.copy()
(num_cols, cat_cols, ord_cols) = col_types(dff)
for col in dff.columns:
    print('For column: ', col + '\n')
    print(dff[col].value_counts(), '\n')

def bar_box(df, col, target='SalePrice'):
    sns.set_style('darkgrid')
    (fig, axes) = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
    order = sorted(df[col].unique())
    sns.countplot(data=df[df[target].notnull()], x=col, ax=axes[0], order=order)
    sns.countplot(data=df[df[target].isnull()], x=col, ax=axes[1], order=order)
    sns.boxplot(data=df, x=col, ax=axes[2], y=target, order=order)
    fig.suptitle('For Feature:  ' + col)
    axes[0].set_title('in Training Set ')
    axes[1].set_title('in Test Set ')
    axes[2].set_title(col + ' --- ' + target)
    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(rotation=90)

def plot_scatter(df, col, target='SalePrice'):
    sns.set_style('darkgrid')
    corr = df[[col, target]].corr()[col][1]
    c = ['red'] if corr >= 0.7 else ['brown'] if corr >= 0.3 else ['lightcoral'] if corr >= 0 else ['blue'] if corr <= -0.7 else ['royalblue'] if corr <= -0.3 else ['lightskyblue']
    (fig, ax) = plt.subplots(figsize=(5, 5))
    sns.scatterplot(x=col, y=target, data=df, c=c, ax=ax)
    ax.set_title('Correlation between ' + col + ' and ' + target + ' is: ' + str(corr.round(4)))

def feature_distribution(df, col, target='SalePrice', test=True):
    sns.set_style('darkgrid')
    if _input0 == True:
        (fig, axes) = plt.subplots(1, 5, figsize=(25, 5))
        sns.kdeplot(data=df[df[target].notnull()], x=col, fill=True, label='Train', ax=axes[0], color='orangered')
        sns.kdeplot(data=df[df[target].isnull()], x=col, fill=True, label='Test', ax=axes[0], color='royalblue')
        axes[0].set_title('Distribution')
        axes[0].legend(loc='best')
        sns.boxplot(data=df[df[target].notnull()], y=col, ax=axes[1], color='orangered')
        sns.boxplot(data=df[df[target].isnull()], y=col, ax=axes[2], color='royalblue')
        axes[2].set_ylim(axes[1].get_ylim())
        axes[1].set_title('Boxplot For Train Data')
        axes[2].set_title('Boxplot For Test Data')
        stats.probplot(df[df[target].notnull()][col], plot=axes[3])
        stats.probplot(df[df[target].isnull()][col], plot=axes[4])
        axes[4].set_ylim(axes[3].get_ylim())
        axes[3].set_title('Probability Plot For Train data')
        axes[4].set_title('Probability Plot For Test data')
        fig.suptitle('For Feature:  ' + col)
    else:
        (fig, axes) = plt.subplots(1, 3, figsize=(18, 6))
        sns.kdeplot(data=df, x=col, fill=True, ax=axes[0], color='orangered')
        sns.boxplot(data=df, y=col, ax=axes[1], color='orangered')
        stats.probplot(df[col], plot=axes[2])
        axes[0].set_title('Distribution')
        axes[1].set_title('Boxplot')
        axes[2].set_title('Probability Plot')
        fig.suptitle('For Feature:  ' + col)
for col in cat_cols:
    bar_box(dff, col)
dff[cat_cols]
dff['Older1945'] = dff['MSSubClass'].apply(lambda x: 1 if x in ['30', '70'] else 0)
dff['Newer1946'] = dff['MSSubClass'].apply(lambda x: 1 if x in ['20', '60', '120', '160'] else 0)
dff['AllStyles'] = dff['MSSubClass'].apply(lambda x: 1 if x in ['20', '90', '190'] else 0)
dff['AllAges'] = dff['MSSubClass'].apply(lambda x: 1 if x in ['40', '45', '50', '75', '90', '150', '190'] else 0)
dff['Pud'] = dff['MSSubClass'].apply(lambda x: 1 if x in ['120', '150', '160', '180'] else 0)
dff['Split'] = dff['MSSubClass'].apply(lambda x: 1 if x in ['80', '85180'] else 0)
dff['MSSubClass'] = dff['MSSubClass'].apply(lambda x: '180' if x == '150' else x)
dff['MSZoning'] = dff['MSZoning'].apply(lambda x: 'R' if x.startswith('R') else x)
dff['North'] = dff['Neighborhood'].apply(lambda x: 1 if x in ['Blmngtn', 'BrDale', 'ClearCr', 'Gilbert', 'Names', 'NoRidge', 'NPkVill', 'NWAmes', 'NoRidge', 'NridgHt', 'Sawyer', 'Somerst', 'StoneBr', 'Veenker', 'NridgHt'] else 0)
dff['South'] = dff['Neighborhood'].apply(lambda x: 1 if x in ['Blueste', 'Edwards', 'Mitchel', 'MeadowV', 'SWISU', 'IDOTRR', 'Timber'] else 0)
dff['Downtown'] = dff['Neighborhood'].apply(lambda x: 1 if x in ['BrkSide', 'Crawfor', 'OldTown', 'CollgCr'] else 0)
dff['East'] = dff['Neighborhood'].apply(lambda x: 1 if x in ['IDOTRR', 'Mitchel'] else 0)
dff['West'] = dff['Neighborhood'].apply(lambda x: 1 if x in ['Edwards', 'NWAmes', 'SWISU', 'Sawyer', 'SawyerW'] else 0)
dff.loc[(dff['Condition1'] == 'Feedr') | (dff['Condition2'] == 'Feedr'), 'StreetDegree'] = 1
dff.loc[(dff['Condition1'] == 'Artery') | (dff['Condition2'] == 'Artery'), 'StreetDegree'] = 2
dff['StreetDegree'] = dff['StreetDegree'].fillna(0, inplace=False)
dff.loc[dff['Condition1'].isin(['RRNn', 'RRNe']) | dff['Condition2'].isin(['RRNn', 'RRNe']), 'RailroadDegree'] = 1
dff.loc[dff['Condition1'].isin(['RRAn', 'RRAe']) | dff['Condition2'].isin(['RRAn', 'RRAe']), 'RailroadDegree'] = 2
dff['RailroadDegree'] = dff['RailroadDegree'].fillna(0, inplace=False)
dff.loc[(dff['Condition1'] == 'PosN') | (dff['Condition2'] == 'PosN'), 'OffsiteFeature'] = 1
dff.loc[(dff['Condition1'] == 'PosA') | (dff['Condition2'] == 'PosA'), 'OffsiteFeature'] = 2
dff['OffsiteFeature'] = dff['OffsiteFeature'].fillna(0, inplace=False)
dff['Norm1'] = dff['Condition1'].apply(lambda x: 1 if x == 'Norm' else 0)
dff['Norm2'] = dff['Condition2'].apply(lambda x: 1 if x == 'Norm' else 0)
dff['Norm'] = dff['Norm1'] + dff['Norm2']
dff = dff.drop(['Norm1', 'Norm2'], axis=1, inplace=False)
lotshape = {'IR3': 1, 'IR2': 2, 'IR1': 3, 'Reg': 4}
landcontour = {'Low': 1, 'HLS': 2, 'Bnk': 3, 'Lvl': 4}
utilities = {'ELO': 1, 'NoSeWa': 2, 'NoSewr': 3, 'AllPub': 4}
landslope = {'Sev': 1, 'Mod': 2, 'Gtl': 3}
general = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
bsmtexposure = {'None': 0, 'No': 0, 'Mn': 1, 'Av': 2, 'Gd': 3}
bsmtfintype = {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
electrical = {'Mix': 1, 'FuseP': 2, 'FuseF': 3, 'FuseA': 4, 'SBrkr': 5}
functional = {'Typ': 1, 'Min1': 2, 'Min2': 3, 'Mod': 4, 'Maj1': 5, 'Maj2': 6, 'Sev': 7, 'Sal': 8}
garagefinish = {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
fence = {'None': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}
dff = dff.replace({'LotShape': lotshape, 'LandContour': landcontour, 'Utilities': utilities, 'LandSlope': landslope, 'BsmtExposure': bsmtexposure, 'BsmtFinType1': bsmtfintype, 'BsmtFinType2': bsmtfintype, 'Electrical': electrical, 'Functional': functional, 'GarageFinish': garagefinish, 'Fence': fence}, inplace=False)
for col in ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']:
    dff[col] = dff[col].replace(general)
dff['BldgType'] = dff['BldgType'].apply(lambda x: '2Fam' if x in ['2fmCon', 'Duplex'] else x)
dff['SaleType'] = dff['SaleType'].apply(lambda x: 'WD' if x.endswith('WD') else x)
dff['SaleType'] = dff['SaleType'].apply(lambda x: 'Contract' if x.startswith('Con') else x)
dff['SaleType'] = dff['SaleType'].apply(lambda x: 'Oth' if x == 'COD' else x)
dff['SaleCondition'] = dff['SaleCondition'].apply(lambda x: 'Abnormal_Adjland' if x in ['Abnorml', 'AdjLand'] else x)
dff['SaleCondition'] = dff['SaleCondition'].apply(lambda x: 'Alloca_Family' if x in ['Alloca', 'Family'] else x)
dff['SaleCondition'] = dff['SaleCondition'].apply(lambda x: 'Other' if x in ['Abnormal_Adjland', 'Alloca_Family'] else x)
dff['GarageType'] = dff['GarageType'].apply(lambda x: 'Carport_None' if x in ['CarPort', 'None'] else x)
dff['GarageType'] = dff['GarageType'].apply(lambda x: 'Basement_2Types' if x in ['Basment', '2Types'] else x)
dff['LotConfig'] = dff['LotConfig'].apply(lambda x: 'CulDSac_FR3' if x in ['CulDSac', 'FR3'] else x)
dff['RoofStyle'] = dff['RoofStyle'].apply(lambda x: 'Other' if x not in ['Gable'] else x)
dff['RoofMatl'] = dff['RoofMatl'].apply(lambda x: 'Other' if x != 'CompShg' else x)
dff['MasVnrType'] = dff['MasVnrType'].apply(lambda x: 'None_BrkCmn' if x in ['None', 'BrkCmn'] else x)
dff['Foundation'] = dff['Foundation'].apply(lambda x: 'BrkTil_Stone' if x in ['BrkTil', 'Stone'] else x)
dff['Foundation'] = dff['Foundation'].apply(lambda x: 'BrkTil_Stone_Slab' if x in ['BrkTil_Stone', 'Slab'] else x)
dff['Foundation'] = dff['Foundation'].apply(lambda x: 'PConc_Wood' if x in ['PConc', 'Wood'] else x)
dff['Heating'] = dff['Heating'].apply(lambda x: 'Other' if x != 'GasA' else x)
for col in num_cols:
    feature_distribution(dff, col)
for col in num_cols:
    plot_scatter(dff, col)
dff[num_cols]
dff['FrontageRatio'] = dff['LotFrontage'] / dff['LotArea']
dff['HQFloor'] = dff['1stFlrSF'] + dff['2ndFlrSF']
dff['FloorAreaRatio'] = dff['GrLivArea'] / dff['LotArea']
dff['TotalArea'] = dff['TotalBsmtSF'] + dff['GrLivArea']
dff['TotalPorch'] = dff['WoodDeckSF'] + dff['OpenPorchSF'] + dff['EnclosedPorch'] + dff['3SsnPorch'] + dff['ScreenPorch']
dff['WeightedBsmtFinSF1'] = dff['BsmtFinSF1'] * dff['BsmtFinType1']
dff['WeightedBsmtFinSF2'] = dff['BsmtFinSF2'] * dff['BsmtFinType2']
dff['WeightedTotalBasement'] = dff['WeightedBsmtFinSF1'] + dff['BsmtFinSF2'] * dff['BsmtFinType2'] + dff['BsmtUnfSF']
dff['TotalFullBath'] = dff['BsmtFullBath'] + dff['FullBath']
dff['TotalHalfBath'] = dff['BsmtHalfBath'] + dff['HalfBath']
dff['TotalBsmtBath'] = dff['BsmtFullBath'] + 0.5 * dff['BsmtHalfBath']
dff['TotalBath'] = dff['TotalFullBath'] + 0.5 * (dff['BsmtHalfBath'] + dff['HalfBath']) + dff['BsmtFullBath'] + 0.5 * dff['BsmtHalfBath']
dff['HasPool'] = dff['PoolArea'].apply(lambda x: 0 if x == 0 else 1)
dff['Has2ndFlr'] = dff['2ndFlrSF'].apply(lambda x: 0 if x == 0 else 1)
dff['HasBsmt'] = dff['TotalBsmtSF'].apply(lambda x: 0 if x == 0 else 1)
dff['HasFireplace'] = dff['Fireplaces'].apply(lambda x: 0 if x == 0 else 1)
dff['HasGarage'] = dff['GarageArea'].apply(lambda x: 0 if x == 0 else 1)
dff['HasLowQual'] = dff['LowQualFinSF'].apply(lambda x: 0 if x == 0 else 1)
dff['HasPorch'] = dff['TotalPorch'].apply(lambda x: 0 if x == 0 else 1)
dff['HasMiscVal'] = dff['MiscVal'].apply(lambda x: 0 if x == 0 else 1)
dff['HasWoodDeck'] = dff['WoodDeckSF'].apply(lambda x: 0 if x == 0 else 1)
dff['HasOpenPorch'] = dff['OpenPorchSF'].apply(lambda x: 0 if x == 0 else 1)
dff['HasEnclosedPorch'] = dff['EnclosedPorch'].apply(lambda x: 0 if x == 0 else 1)
dff['Has3SsnPorch'] = dff['3SsnPorch'].apply(lambda x: 0 if x == 0 else 1)
dff['HasScreenPorch'] = dff['ScreenPorch'].apply(lambda x: 0 if x == 0 else 1)
dff['TotalPorchType'] = dff['HasWoodDeck'] + dff['HasOpenPorch'] + dff['HasEnclosedPorch'] + dff['Has3SsnPorch'] + dff['HasScreenPorch']
dff['TotalPorchType'] = dff['TotalPorchType'].apply(lambda x: 3 if x >= 3 else x)
dff['RestorationAge'] = dff['YearRemodAdd'] - dff['YearBuilt']
dff['RestorationAge'] = dff['RestorationAge'].apply(lambda x: 0 if x < 0 else x)
dff['HasRestoration'] = dff['RestorationAge'].apply(lambda x: 0 if x == 0 else 1)
dff['YearAfterRestoration'] = dff['YrSold'] - dff['YearRemodAdd']
dff['YearAfterRestoration'] = dff['YearAfterRestoration'].apply(lambda x: 0 if x < 0 else x)
dff['BuildAge'] = dff['YrSold'] - dff['YearBuilt']
dff['BuildAge'] = dff['BuildAge'].apply(lambda x: 0 if x < 0 else x)
dff['IsNewHouse'] = dff['BuildAge'].apply(lambda x: 1 if x == 0 else 0)

def year_map(year):
    year = 1 if year <= 1895 else 2 if year <= 1916 else 3 if year <= 1919 else 4 if year <= 1929 else 5 if year <= 1941 else 6 if year <= 1945 else 7 if year <= 1964 else 8 if year <= 1980 else 9 if year <= 1991 else 10 if year < 2008 else 11
    return year
dff['YearBuilt_bins'] = dff['YearBuilt'].apply(lambda year: year_map(year))
dff['YearRemodAdd_bins'] = dff['YearRemodAdd'].apply(lambda year: year_map(year))
dff['GarageYrBlt_bins'] = dff['GarageYrBlt'].apply(lambda year: year_map(year))
dff['YrSold'] = dff['YrSold'].astype(str)
dff['MoSold'] = dff['MoSold'].astype(str)
dff['Season'] = dff['MoSold'].apply(lambda x: 'Winter' if x in ['12', '1', '2'] else 'Spring' if x in ['3', '4', '5'] else 'Summer' if x in ['6', '7', '8'] else 'Fall')
dff['OverallValue'] = dff['OverallQual'] * dff['OverallCond']
dff['ExterValue'] = dff['ExterQual'] * dff['ExterCond']
dff['BsmtValue'] = (dff['BsmtQual'] + dff['BsmtFinType1'] + dff['BsmtFinType2']) * dff['BsmtCond'] / 2
dff['KitchenValue'] = dff['KitchenAbvGr'] * dff['KitchenQual']
dff['FireplaceValue'] = dff['Fireplaces'] * dff['FireplaceQu']
dff['GarageValue'] = dff['GarageQual'] * dff['GarageCond']
dff['TotalValue'] = dff['OverallValue'] + dff['ExterValue'] + dff['BsmtValue'] + dff['KitchenValue'] + dff['FireplaceValue'] + dff['GarageValue'] + dff['HeatingQC'] + dff['Utilities'] + dff['Electrical'] - dff['Functional'] + dff['PoolQC']
dff['TotalQual'] = dff['OverallQual'] + dff['ExterQual'] + dff['BsmtQual'] + dff['KitchenQual'] + dff['FireplaceQu'] + dff['GarageQual'] + dff['HeatingQC'] + dff['PoolQC']
dff['TotalCond'] = dff['OverallCond'] + dff['ExterCond'] + dff['BsmtCond'] + dff['GarageCond']
dff['TotalQualCond'] = dff['TotalQual'] + dff['TotalCond']
dff['BsmtSFxValue'] = dff['TotalBsmtSF'] * dff['BsmtValue']
dff['BsmtSFxQual'] = dff['TotalBsmtSF'] * dff['BsmtQual']
dff['TotalAreaXOverallValue'] = dff['TotalArea'] * dff['OverallValue']
dff['TotalAreaXOverallQual'] = dff['TotalArea'] * dff['OverallQual']
dff['GarageAreaXGarageValue'] = dff['GarageArea'] * dff['GarageValue']
dff['GarageAreaXGarageQual'] = dff['GarageArea'] * dff['GarageQual']
dff2 = dff.copy()
(num_cols2, cat_cols2, ord_cols2) = col_types(dff2)
for col in ord_cols2:
    bar_box(dff, col)
dff2['LotShape'] = dff2['LotShape'].apply(lambda x: 1 if x in [1, 2] else 2 if x == 3 else 3)
dff2['LandSlope'] = dff2['LandSlope'].apply(lambda x: 1 if x in [1, 2] else 2 if x == 3 else 3)
dff2['OverallCond'] = dff2['OverallCond'].apply(lambda x: 1 if x in [1, 2, 3] else x - 1)
dff2['OverallQual'] = dff2['OverallQual'].apply(lambda x: 1 if x in [1, 2] else x - 1)
dff2['ExterCond'] = dff2['ExterCond'].apply(lambda x: 1 if x in [1, 2] else 2 if x == 3 else 3)
dff2['BsmtQual'] = dff2['BsmtQual'].apply(lambda x: 0 if x in [0, 1, 2] else 1 if x == 3 else 2 if x == 4 else 3)
dff2['BsmtCond'] = dff2['BsmtCond'].apply(lambda x: 0 if x in [0, 1, 2] else 1 if x == 3 else 2)
dff2['BsmtFinType1'] = dff2['BsmtFinType1'].apply(lambda x: 1 if x in [1, 2, 3, 4, 5] else 2 if x == 6 else x)
dff2['BsmtFinType2'] = dff2['BsmtFinType2'].apply(lambda x: 1 if x in [1, 2, 3, 4, 5] else 2 if x == 6 else x)
dff2['HeatingQC'] = dff2['HeatingQC'].apply(lambda x: 1 if x in [1, 2] else 2 if x in [3, 4] else 3)
dff2['Electrical'] = dff2['Electrical'].apply(lambda x: 1 if x in [1, 2] else x - 3)
dff2['BsmtFullBath'] = dff2['BsmtFullBath'].apply(lambda x: 2 if x >= 2 else x)
dff2['FullBath'] = dff2['FullBath'].apply(lambda x: 1 if x <= 1 else 3 if x >= 3 else x)
dff2['HalfBath'] = dff2['HalfBath'].apply(lambda x: 1 if x >= 1 else 0)
dff2['BedroomAbvGr'] = dff2['BedroomAbvGr'].apply(lambda x: 1 if x <= 1 else 5 if x >= 5 else x)
dff2['KitchenAbvGr'] = dff2['KitchenAbvGr'].apply(lambda x: 1 if x <= 1 else 2 if x >= 2 else x)
dff2['TotRmsAbvGrd'] = dff2['TotRmsAbvGrd'].apply(lambda x: 3 if x <= 4 else 10 if x >= 11 else x - 1)
dff2['Functional'] = dff2['Functional'].apply(lambda x: 1 if x == 1 else 2)
dff2['Fireplaces'] = dff2['Fireplaces'].apply(lambda x: 2 if x >= 2 else x)
dff2['GarageCars'] = dff2['GarageCars'].apply(lambda x: 3 if x >= 3 else x)
dff2['GarageQual'] = dff2['GarageQual'].apply(lambda x: 1 if x <= 2 else 2 if x == 3 else 3)
dff2['GarageCond'] = dff2['GarageCond'].apply(lambda x: 1 if x <= 2 else 2)
dff2['Fence'] = dff2['Fence'].apply(lambda x: 1 if x in [1, 3] else x)
dff3 = dff2.copy()
dff3
for col in cat_cols2:
    print(col, dff3[col].value_counts().size)
target_encoding = ['MSSubClass', 'Neighborhood', 'Exterior1st', 'Exterior2nd', 'Condition1', 'Condition2', 'HouseStyle']
for col in target_encoding:
    feature_name = col + 'Rank'
    dff3.loc[:, feature_name] = dff3[col].map(dff3.groupby(col).SalePrice.median())
    dff3.loc[:, feature_name] = dff3.loc[:, feature_name].rank(method='dense')
dff3['Exterior'] = np.where(dff3['Exterior1st'] != dff3['Exterior2nd'], 'Mixed', dff3['Exterior1st'])
dff3['No2ndExt'] = dff3['Exterior'].apply(lambda x: 0 if x == 'Mixed' else 1)
drop_cols = ['MSSubClass', 'Neighborhood', 'Condition1', 'Condition2', 'Exterior1st', 'Exterior2nd', 'PoolArea', 'PoolQC', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'LowQualFinSF', 'MiscVal', '2ndFlrSF', 'HouseStyle', 'YrSold', 'MoSold', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'Exterior', 'Utilities', 'Street']
dff3 = dff3.drop(drop_cols, axis=1, inplace=False)
(num_, cat_, ord_) = col_types(dff3)

def prep_data(df, cat_cols, target):
    dummies = pd.get_dummies(df[cat_cols], drop_first=True)
    data = pd.concat([df, dummies], axis=1).drop(cat_cols, axis=1)
    _input1 = data[data[target].notnull()]
    _input0 = data[data[target].isnull()]
    return (_input1, _input0)
(_input1, _input0) = prep_data(dff3, cat_, 'SalePrice')
target = 'SalePrice'
predictors = [x for x in _input1.columns if x not in ['Id', 'SalePrice']]
train_skew = []
test_skew = []
check_cols = [x for x in num_ if not x.endswith('Rank')]
for col in check_cols:
    train_skew.append(_input1[col].skew())
    test_skew.append(_input0[col].skew())
skew_df = pd.DataFrame({'Feature': check_cols, 'TrainSkewness': train_skew, 'TestSkewness': test_skew})
skewed = skew_df[skew_df.TrainSkewness.abs() >= 0.5]
skewed
train_skew_yeoj = []
test_skew_yeoj = []
for col in skewed.Feature.tolist():
    (_input1[col], fitted_lambda) = stats.yeojohnson(_input1[col])
    _input0[col] = stats.yeojohnson(_input0[col], fitted_lambda)
    train_skew_yeoj.append(_input1[col].skew())
    test_skew_yeoj.append(_input0[col].skew())
skewed['TrainSkewness_AfterYeoJohnson'] = train_skew_yeoj
skewed['TestSkewness_AfterYeoJohnson'] = test_skew_yeoj
skewed
high_skew = skewed[skewed.TrainSkewness_AfterYeoJohnson.abs() > 1].Feature.tolist()
print(high_skew)
_input1 = _input1.drop(high_skew, axis=1, inplace=False)
_input0 = _input0.drop(high_skew, axis=1, inplace=False)
feature_distribution(_input1, target, test=False)
_input1[target].skew()
_input1[target] = np.log1p(_input1[target])
feature_distribution(_input1, target, test=False)
_input1[target].skew()
target = 'SalePrice'
predictors = [x for x in _input1.columns if x not in ['Id', 'SalePrice']]
scaler = RobustScaler()
_input1[predictors] = scaler.fit_transform(_input1[predictors])
_input0[predictors] = scaler.transform(_input0[predictors])
X_train = _input1[predictors]
y_train = _input1[target]
X_test = _input0[predictors]
print(X_train.shape)
print(X_test.shape)
selector = VarianceThreshold(0.01)