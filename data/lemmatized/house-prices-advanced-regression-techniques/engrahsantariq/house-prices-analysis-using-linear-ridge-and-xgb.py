import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as cf
import plotly.express as px
cf.go_offline()
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import shapiro
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 25)
import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input2 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
_input1[_input1['BsmtFinSF1'] == 0][['BsmtFinSF1', 'BsmtFinSF2']]
for i in [_input1, _input0]:
    i.info()
_input1[['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']]
print('The length of Test Data given in csv file is ', len(_input0.columns), '\nThe length of Train Data in csv file is ', len(_input1.columns))
_input1.describe()
_input0.describe()
_input1['MoSold'].value_counts()
_input0['MoSold'].value_counts()
plt.figure(figsize=(14, 8))
sns.heatmap(_input1.isnull(), yticklabels=False, cmap='viridis')
plt.figure(figsize=(14, 8))
sns.heatmap(_input0.isnull(), yticklabels=False, cmap='viridis')
na_data = _input1.isnull().sum() / len(_input1) * 100
na_data = na_data.drop(na_data[na_data == 0].index).sort_values(ascending=False)[:15]
missing_data_analysis = pd.DataFrame({'Missing Data': na_data})
missing_data_analysis
all_testdata_na = _input0.isnull().sum() / len(_input0) * 100
all_testdata_na = all_testdata_na.drop(all_testdata_na[all_testdata_na == 0].index).sort_values(ascending=False)[:15]
missing_testdata = pd.DataFrame({'Missing Data': all_testdata_na})
missing_testdata
(fig, ax) = plt.subplots(figsize=(14, 8))
sns.barplot(x=na_data.index, y=na_data)
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Ratio of Missing Values')
plt.title('Missing Values in Training Set (Ratio)')
(fig, ax) = plt.subplots(figsize=(14, 8))
sns.barplot(x=all_testdata_na.index, y=all_testdata_na)
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Ratio of Missing Values')
plt.title('Missing Values in Test Set (Ratio)')
for column in _input1.columns:
    print(column + ':', _input1[column].isnull().sum(), _input1[column].dtype)
_input1.isnull().sum().idxmax()
_input1[_input1['GarageYrBlt'].isnull() & (_input1['YearBuilt'] > 1800)]
for column in _input0.columns:
    print(column + ':', _input0[column].isnull().sum(), 'dtype: ', _input0[column].dtype)
_input0.isnull().sum().idxmax()
_input1['PoolQC'].isnull().sum()
plt.figure(figsize=(14, 8))
sns.heatmap(_input1.corr())
plt.figure(figsize=(14, 8))
_input1.corr()['SalePrice'].sort_values()[:-1].plot(kind='bar')
_input1.corr()['SalePrice'].sort_values(ascending=False)[1:]
_input1['SalePrice'].sort_values(ascending=True)[:10]
plt.figure(figsize=(14, 8))
sns.distplot(_input1['SalePrice'], kde=True, bins=50)
_input1[_input1['SalePrice'] > 600000]
_input1[_input1['SalePrice'] < 50000]
plt.figure(figsize=(14, 8))
sns.distplot(_input1['SalePrice'], kde=True, bins=50)
(fig, axes) = plt.subplots(1, 2, figsize=(18, 8))
sns.scatterplot(x='SalePrice', y='YearBuilt', data=_input1, ax=axes[0])
sns.scatterplot(x='SalePrice', y='YearRemodAdd', data=_input1, ax=axes[1])
axes[0].set_title('SalePrice vs YearBuilt')
axes[1].set_title('SalePrice vs Year Remodeled')
plt.figure(figsize=(14, 8))
stats.probplot(_input1['SalePrice'], plot=plt, dist='norm')
_input1['LogSalePrice'] = np.log(_input1['SalePrice'])
plt.figure(figsize=(14, 8))
sns.distplot(_input1['LogSalePrice'], kde=True, bins=50)
plt.figure(figsize=(14, 8))
stats.probplot(_input1['LogSalePrice'], plot=plt, dist='norm')
sns.boxplot(data=_input1, y='LogSalePrice')
_input1['LogSalePrice'].describe()
(fig, ax) = plt.subplots(1, 2, figsize=(14, 8))
sns.countplot(data=_input1, x='GarageCars', ax=ax[0])
sns.countplot(data=_input0, x='GarageCars', ax=ax[1])
ax[0].set_title('Number Car Garage Training Set')
ax[1].set_title('Number Car Garage Test Set')
print(_input1['GarageCars'].value_counts())
print(_input0['GarageCars'].value_counts())
print(_input1['GarageCars'].isnull().sum())
print(_input0['GarageCars'].isnull().sum())
print(_input1['GrLivArea'].value_counts())
print(_input0['GrLivArea'].value_counts())
print(_input1['GrLivArea'].isnull().sum())
print(_input0['GrLivArea'].isnull().sum())
(fig, ax) = plt.subplots(1, 2, figsize=(14, 8))
sns.histplot(x=_input1['GrLivArea'], ax=ax[0], bins=50)
sns.histplot(x=_input0['GrLivArea'], ax=ax[1], bins=50)
ax[0].set_title('Above Ground Living Area Training Set')
ax[1].set_title('Above Ground Living Area Testing Set')
print('mode: ', _input1['SalePrice'].mode())
print('median: ', _input1['SalePrice'].median())
print('mean: ', _input1['SalePrice'].mean())
print('min: ', _input1['SalePrice'].min())
print('max: ', _input1['SalePrice'].max())
print(_input1['SalePrice'].count())
for i in _input1.columns:
    print(i + ': ', _input1[i].dtypes)
print('train_data\n', _input1.dtypes.value_counts())
print('test_data\n', _input0.dtypes.value_counts())
print(_input1['Street'].value_counts())
print(_input1['Street'].isnull().sum())
print(_input1['Alley'].value_counts())
print(_input1['Alley'].isnull().sum())
print(_input0['Alley'].value_counts())
print(_input0['Alley'].isnull().sum())
for column in _input1.columns:
    if _input1[column].isnull().sum() / len(_input1) > 0.5:
        print(column)
_input1 = _input1.drop(['Alley', 'Fence', 'FireplaceQu', 'MiscFeature'], axis=1)
for column in _input0.columns:
    if _input0[column].isnull().sum() / len(_input0) > 0.5:
        print(column)
_input0 = _input0.drop(['FireplaceQu', 'Alley', 'Fence', 'MiscFeature'], axis=1)
_input1['LotFrontage'] = _input1.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
_input0['LotFrontage'] = _input0.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
plt.figure(figsize=(14, 8))
sns.heatmap(_input1.isnull(), yticklabels=False, cmap='viridis')
plt.figure(figsize=(14, 8))
sns.heatmap(_input0.isnull(), yticklabels=False, cmap='viridis')
plt.figure(figsize=(20, 12))
sns.heatmap(_input1.corr(), annot=False)
plt.figure(figsize=(14, 8))
sns.countplot(data=_input1, x=_input1['Heating'])
_input1['Heating'].value_counts()
(fig, ax) = plt.subplots(1, 2, figsize=(14, 8))
sns.countplot(data=_input1, x='GarageCars', hue='PavedDrive', ax=ax[0])
sns.countplot(data=_input1, x='GarageQual', ax=ax[1])
ax[0].set_title('#Cars Garage vs. Paved/Not')
ax[1].set_title('Garage Quality')
(fig, ax) = plt.subplots(1, 2, figsize=(14, 8))
sns.countplot(data=_input0, x='GarageCars', hue='PavedDrive', ax=ax[0])
sns.countplot(data=_input0, x='GarageQual', ax=ax[1])
ax[0].set_title('#Cars Garage vs. Paved/Not...Test set')
ax[1].set_title('Garage Quality...Test set')
plt.figure(figsize=(14, 8))
sns.countplot(data=_input0, x='MSZoning', hue='Neighborhood')
plt.title('MSZoning vs Neighborhood Test set')
plt.legend(loc=1)
plt.figure(figsize=(14, 8))
_input1.corr()['SalePrice'].sort_values()[22:-1].plot(kind='bar')
(fig, ax) = plt.subplots(1, 2, figsize=(14, 8))
sns.countplot(data=_input1, x='Fireplaces', ax=ax[0])
sns.countplot(data=_input0, x='Fireplaces', ax=ax[1])
ax[0].set_title('# Fireplaces Training set')
ax[1].set_title('# Fireplaces Test set')
print(_input1['GarageQual'].isnull().sum())
print(_input0['GarageQual'].isnull().sum())
print('OverallQual_Train', _input1['OverallQual'].isnull().sum())
print('OverallQual_Test', _input0['OverallQual'].isnull().sum())
print('GarageQual_Train', _input1['GarageQual'].isnull().sum())
print('GarageQual_Test', _input0['GarageQual'].isnull().sum())
print('GrLivArea_Train', _input1['GrLivArea'].isnull().sum())
print('GrLivArea_Test', _input0['GrLivArea'].isnull().sum())
print('GarageCars_Train', _input1['GarageCars'].isnull().sum())
print('GarageCars_Test', _input0['GarageCars'].isnull().sum())
print('TotalBsmtSF_Train', _input1['TotalBsmtSF'].isnull().sum())
print('TotalBmstSF_Test', _input0['TotalBsmtSF'].isnull().sum())
print('1stFlrSF_Train', _input1['1stFlrSF'].isnull().sum())
print('1stFlrSF_test', _input0['1stFlrSF'].isnull().sum())
print('FullBath', _input1['FullBath'].isnull().sum())
print('FullBath', _input0['FullBath'].isnull().sum())
print('TotalRmsAbvGrd_Train', _input1['TotRmsAbvGrd'].isnull().sum())
print('TotalRmsAbvGrd_Test', _input0['TotRmsAbvGrd'].isnull().sum())
print('YearBuilt_Train', _input1['YearBuilt'].isnull().sum())
print('YearBuilt_Test', _input0['YearBuilt'].isnull().sum())
print('YearRemodeled_Train', _input1['YearRemodAdd'].isnull().sum())
print('YearRemodeled_Test', _input0['YearRemodAdd'].isnull().sum())
_input1['GarageQual'] = _input1['GarageQual'].fillna('None')
_input0['GarageQual'] = _input0['GarageQual'].fillna('None')
_input0['MSZoning'] = _input0.groupby('Neighborhood')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
_input0[_input0['TotalBsmtSF'].isnull()]
_input0['GarageType'].isnull().sum()
_input0[_input0['GarageType'].isnull()]
_input1['BsmtQual'].isnull().sum()
df = _input1[_input1['BsmtQual'].isnull()]
df[df.columns[30:35]].head()
_input1['BsmtCond'] = _input1['BsmtCond'].fillna('None')
_input1['BsmtExposure'] = _input1['BsmtExposure'].fillna('None')
_input1['BsmtFinType1'] = _input1['BsmtFinType1'].fillna('None')
_input1['BsmtFinType2'] = _input1['BsmtFinType2'].fillna('None')
print(_input1['BsmtCond'].isnull().sum())
print(_input1['BsmtExposure'].isnull().sum())
print(_input1['BsmtFinType1'].isnull().sum())
print(_input1['BsmtFinType2'].isnull().sum())
_input0['BsmtCond'] = _input0['BsmtCond'].fillna('None')
_input0['BsmtExposure'] = _input0['BsmtExposure'].fillna('None')
_input0['BsmtFinType1'] = _input0['BsmtFinType1'].fillna('None')
_input0['BsmtFinType2'] = _input0['BsmtFinType2'].fillna('None')
print(_input0['BsmtCond'].isnull().sum())
print(_input0['BsmtExposure'].isnull().sum())
print(_input0['BsmtFinType1'].isnull().sum())
print(_input0['BsmtFinType2'].isnull().sum())
plt.figure(figsize=(14, 8))
sns.heatmap(_input0.isnull(), yticklabels=False, cmap='viridis')
plt.figure(figsize=(14, 8))
sns.heatmap(_input1.isnull(), yticklabels=False, cmap='viridis')
_input1['MasVnrType'].isnull().sum()
_input1[_input1['MasVnrType'].isnull()]
plt.figure(figsize=(14, 8))
sns.countplot(data=_input1, x='Exterior1st', hue='MasVnrType')
plt.legend(loc=1)
plt.figure(figsize=(14, 8))
sns.countplot(data=_input1, x='Exterior2nd', hue='MasVnrType')
plt.legend(loc=1)

def mason_veneer(cols):
    MasonType = cols[0]
    Exterior1 = cols[1]
    if pd.isnull(MasonType):
        if Exterior1 == 'Vinyl':
            return 'None'
        elif Exterior1 == 'HdBoard':
            return 'BrkFace'
        else:
            return 'None'
    else:
        return MasonType
_input1['MasVnrType'] = _input1[['MasVnrType', 'Exterior1st']].apply(mason_veneer, axis=1)
_input1[_input1['MasVnrType'] == 'None'][['MasVnrType', 'MasVnrArea']]
plt.figure(figsize=(14, 8))
sns.heatmap(_input1.isnull(), yticklabels=False, cmap='viridis')
plt.figure(figsize=(14, 8))
sns.heatmap(_input0.isnull(), yticklabels=False, cmap='viridis')
_input1['MasVnrArea'] = _input1['MasVnrArea'].fillna(0)
_input0['MasVnrArea'] = _input0['MasVnrArea'].fillna(0)
plt.figure(figsize=(14, 8))
sns.heatmap(_input1.isnull(), yticklabels=False, cmap='viridis')
plt.figure(figsize=(14, 8))
sns.heatmap(_input0.isnull(), yticklabels=False, cmap='viridis')
for column in _input0.columns:
    print(column + ':', _input0[column].isnull().sum(), _input0[column].dtype)
_input1 = _input1.drop(['PoolArea', 'PoolQC'], axis=1)
_input0 = _input0.drop(['PoolArea', 'PoolQC'], axis=1)
_input1 = _input1.drop(['MSSubClass', 'MSZoning', 'LotArea', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope'], axis=1)
_input0 = _input0.drop(['MSSubClass', 'MSZoning', 'LotArea', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope'], axis=1)
_input1 = _input1.drop('Condition2', axis=1)
_input0 = _input0.drop('Condition2', axis=1)
_input1 = _input1.drop(['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF'], axis=1)
_input0 = _input0.drop(['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF'], axis=1)
_input1 = _input1.drop(['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageArea', 'GarageQual', 'GarageCond'], axis=1)
_input0 = _input0.drop(['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageArea', 'GarageQual', 'GarageCond'], axis=1)
_input1 = _input1.drop(['1stFlrSF', '2ndFlrSF', 'LowQualFinSF'], axis=1)
_input0 = _input0.drop(['1stFlrSF', '2ndFlrSF', 'LowQualFinSF'], axis=1)
_input1 = _input1.drop(['RoofStyle', 'RoofMatl', 'BsmtFinType2'], axis=1)
_input0 = _input0.drop(['RoofStyle', 'RoofMatl', 'BsmtFinType2'], axis=1)
_input1 = _input1.drop(['Exterior2nd', 'BsmtExposure', 'Electrical', 'MiscVal'], axis=1)
_input0 = _input0.drop(['Exterior2nd', 'BsmtExposure', 'Electrical', 'MiscVal'], axis=1)
_input1 = _input1.drop('Functional', axis=1)
_input0 = _input0.drop('Functional', axis=1)
_input1 = _input1.drop(['MasVnrType', 'MasVnrArea'], axis=1)
_input0 = _input0.drop(['MasVnrType', 'MasVnrArea'], axis=1)
_input1 = _input1.drop('BsmtFinType1', axis=1)
_input0 = _input0.drop('BsmtFinType1', axis=1)
_input1 = _input1.drop('SaleType', axis=1)
_input0 = _input0.drop('SaleType', axis=1)
_input1['TotHalfBaths'] = _input1['BsmtHalfBath'] + _input1['HalfBath']
_input1['TotalFullBaths'] = _input1['BsmtFullBath'] + _input1['FullBath']
_input0['TotHalfBaths'] = _input0['BsmtHalfBath'] + _input0['HalfBath']
_input0['TotalFullBaths'] = _input0['BsmtFullBath'] + _input0['FullBath']
_input1 = _input1.drop(['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath'], axis=1)
_input0 = _input0.drop(['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath'], axis=1)
_input0['TotalBsmtSF'] = _input0['TotalBsmtSF'].fillna(round(_input0['TotalBsmtSF'].mean(), 0))
_input0['GarageCars'] = _input0['GarageCars'].fillna(0)
_input0['Exterior1st'] = _input0['Exterior1st'].fillna(_input0['Exterior1st'].mode()[0])
_input0['BsmtQual'] = _input0['BsmtQual'].fillna('None')
_input1['BsmtQual'] = _input1['BsmtQual'].fillna('None')
_input0['KitchenQual'] = _input0['KitchenQual'].fillna(_input0['KitchenQual'].mode()[0])
_input1.groupby('Neighborhood')['SalePrice'].mean().sort_values(ascending=False)
plt.figure(figsize=(14, 8))
sns.heatmap(_input1.isnull(), yticklabels=False, cmap='viridis')
plt.figure(figsize=(14, 8))
sns.heatmap(_input0.isnull(), yticklabels=False, cmap='viridis')
_input1['TotHalfBaths'] = _input1['TotHalfBaths'].fillna(_input1['TotHalfBaths'].mode()[0])
_input0['TotHalfBaths'] = _input0['TotHalfBaths'].fillna(_input0['TotHalfBaths'].mode()[0])
_input1['TotalFullBaths'] = _input1['TotalFullBaths'].fillna(_input1['TotalFullBaths'].mode()[0])
_input0['TotalFullBaths'] = _input0['TotalFullBaths'].fillna(_input0['TotalFullBaths'].mode()[0])
_input1['HeatingQC'].value_counts()
heatingqc_mapping = {'Ex': 0, 'TA': 1, 'Gd': 2, 'Fa': 3, 'Po': 4}
_input1['HeatingQC'] = _input1['HeatingQC'].map(heatingqc_mapping)
_input0['HeatingQC'] = _input0['HeatingQC'].map(heatingqc_mapping)
_input1.head()
print(_input1['HouseStyle'].value_counts())
print(_input0['HouseStyle'].value_counts())
housestyle_mapping = {'1Story': 0, '2Story': 1, '1.5Fin': 2, 'SLvl': 3, 'SFoyer': 4, '1.5Unf': 5, '2.5Unf': 6, '2.5Fin': 7}
_input1['HouseStyle'] = _input1['HouseStyle'].map(housestyle_mapping)
_input0['HouseStyle'] = _input0['HouseStyle'].map(housestyle_mapping)
print(_input1['BsmtCond'].value_counts())
print(_input0['BsmtCond'].value_counts())
bsmtcond_mapping = {'None': 0, 'TA': 1, 'Gd': 2, 'Fa': 3, 'Po': 4}
_input1['BsmtCond'] = _input1['BsmtCond'].map(bsmtcond_mapping)
_input0['BsmtCond'] = _input0['BsmtCond'].map(bsmtcond_mapping)
_input1['ExterCond'].value_counts()
extercond_mapping = {'TA': 0, 'Gd': 1, 'Fa': 2, 'Ex': 3, 'Po': 4}
_input1['ExterCond'] = _input1['ExterCond'].map(extercond_mapping)
_input0['ExterCond'] = _input0['ExterCond'].map(extercond_mapping)
print(_input1['ExterCond'].value_counts())
print(_input0['ExterCond'].value_counts())
neighborhood_mapping = {'NridgHt': 0, 'StoneBr': 1, 'NoRidge': 2, 'Timber': 3, 'Veenker': 4, 'Somerst': 5, 'ClearCr': 6, 'Crawfor': 7, 'CollgCr': 8, 'Blmngtn': 9, 'Gilbert': 10, 'NWAmes': 11, 'SawyerW': 12, 'Mitchel': 13, 'NAmes': 14, 'NPkVill': 15, 'SWISU': 16, 'Blueste': 17, 'Sawyer': 18, 'OldTown': 19, 'Edwards': 20, 'BrkSide': 21, 'BrDale': 22, 'IDOTRR': 23, 'MeadowV': 24}
_input1['Neighborhood'] = _input1['Neighborhood'].map(neighborhood_mapping)
_input0['Neighborhood'] = _input0['Neighborhood'].map(neighborhood_mapping)
ac_mapping = {'N': 0, 'Y': 1}
_input1['CentralAir'] = _input1['CentralAir'].map(ac_mapping)
_input0['CentralAir'] = _input0['CentralAir'].map(ac_mapping)
heat_mapping = {'Floor': 0, 'OthW': 1, 'Wall': 2, 'Grav': 3, 'GasW': 4, 'GasA': 5}
_input1['Heating'] = _input1['Heating'].map(heat_mapping)
_input0['Heating'] = _input0['Heating'].map(heat_mapping)
condition_mapping = {'Norm': 0, 'Feedr': 1, 'Artery': 2, 'RRAn': 3, 'PosN': 4, 'RRAe': 5, 'PosA': 6, 'RRNn': 7, 'RRNe': 8}
_input1['Condition1'] = _input1['Condition1'].map(condition_mapping)
_input0['Condition1'] = _input0['Condition1'].map(condition_mapping)
bldg_mapping = {'1Fam': 0, 'TwnhsE': 1, 'Duplex': 2, 'Twnhs': 3, '2fmCon': 4}
_input1['BldgType'] = _input1['BldgType'].map(bldg_mapping)
_input0['BldgType'] = _input0['BldgType'].map(bldg_mapping)
sale_condition_mapping = {'Normal': 0, 'Partial': 1, 'Abnorml': 2, 'Family': 3, 'Alloca': 4, 'AdjLand': 5}
_input1['SaleCondition'] = _input1['SaleCondition'].map(sale_condition_mapping)
_input0['SaleCondition'] = _input0['SaleCondition'].map(sale_condition_mapping)
_input1['Exterior1st'].value_counts()
_input1['Exterior1st'] = _input1['Exterior1st'].replace(['WdShing', 'Stucco', 'AsbShng', 'BrkComm', 'Stone', 'AsphShn', 'ImStucc', 'CBlock'], 'Other')
_input0['Exterior1st'] = _input0['Exterior1st'].replace(['WdShing', 'Stucco', 'AsbShng', 'BrkComm', 'Stone', 'AsphShn', 'ImStucc', 'CBlock'], 'Other')
_input1['Exterior1st'].value_counts()
exterior_mapping = {'VinylSd': 0, 'MetalSd': 1, 'HdBoard': 2, 'Wd Sdng': 3, 'Plywood': 4, 'CemntBd': 5, 'BrkFace': 6, 'Other': 7}
_input1['Exterior1st'] = _input1['Exterior1st'].map(exterior_mapping)
_input0['Exterior1st'] = _input0['Exterior1st'].map(exterior_mapping)
exterqual_mapping = {'TA': 0, 'Gd': 1, 'Ex': 2, 'Fa': 3}
_input1['ExterQual'] = _input1['ExterQual'].map(exterqual_mapping)
_input0['ExterQual'] = _input0['ExterQual'].map(exterqual_mapping)
paved_mapping = {'P': 0, 'N': 1, 'Y': 2}
_input1['PavedDrive'] = _input1['PavedDrive'].map(paved_mapping)
_input0['PavedDrive'] = _input0['PavedDrive'].map(paved_mapping)
found_mapping = {'PConc': 0, 'CBlock': 1, 'BrkTil': 2, 'Slab': 3, 'Stone': 4, 'Wood': 5}
_input1['Foundation'] = _input1['Foundation'].map(found_mapping)
_input0['Foundation'] = _input0['Foundation'].map(found_mapping)
bsmtqual_mapping = {'TA': 0, 'Gd': 1, 'Ex': 2, 'None': 3, 'Fa': 4}
_input1['BsmtQual'] = _input1['BsmtQual'].map(bsmtqual_mapping)
_input0['BsmtQual'] = _input0['BsmtQual'].map(bsmtqual_mapping)
kitchqual_mapping = {'TA': 0, 'Gd': 1, 'Ex': 2, 'Fa': 3}
_input1['KitchenQual'] = _input1['KitchenQual'].map(kitchqual_mapping)
_input0['KitchenQual'] = _input0['KitchenQual'].map(kitchqual_mapping)
_input1['Year_built_range'] = pd.cut(_input1['YearBuilt'], 10)
_input1['Year_built_range'].value_counts()
_input1.loc[_input1['YearBuilt'] <= 1899.6, 'YearBuilt'] = 0
_input1.loc[(_input1['YearBuilt'] > 1899.6) & (_input1['YearBuilt'] <= 1913.4), 'YearBuilt'] = 1
_input1.loc[(_input1['YearBuilt'] > 1913.4) & (_input1['YearBuilt'] <= 1927.2), 'YearBuilt'] = 2
_input1.loc[(_input1['YearBuilt'] > 1927.2) & (_input1['YearBuilt'] <= 1941), 'YearBuilt'] = 3
_input1.loc[(_input1['YearBuilt'] > 1941) & (_input1['YearBuilt'] <= 1954.8), 'YearBuilt'] = 4
_input1.loc[(_input1['YearBuilt'] > 1954.8) & (_input1['YearBuilt'] <= 1968.6), 'YearBuilt'] = 5
_input1.loc[(_input1['YearBuilt'] > 1968.6) & (_input1['YearBuilt'] <= 1982.4), 'YearBuilt'] = 6
_input1.loc[(_input1['YearBuilt'] > 1982.4) & (_input1['YearBuilt'] <= 1996.2), 'YearBuilt'] = 7
_input1.loc[_input1['YearBuilt'] > 1996.2, 'YearBuilt'] = 8
_input1['YearBuilt'].value_counts()
_input0.loc[_input0['YearBuilt'] <= 1899.6, 'YearBuilt'] = 0
_input0.loc[(_input0['YearBuilt'] > 1899.6) & (_input0['YearBuilt'] <= 1913.4), 'YearBuilt'] = 1
_input0.loc[(_input0['YearBuilt'] > 1913.4) & (_input0['YearBuilt'] <= 1927.2), 'YearBuilt'] = 2
_input0.loc[(_input0['YearBuilt'] > 1927.2) & (_input0['YearBuilt'] <= 1941), 'YearBuilt'] = 3
_input0.loc[(_input0['YearBuilt'] > 1941) & (_input0['YearBuilt'] <= 1954.8), 'YearBuilt'] = 4
_input0.loc[(_input0['YearBuilt'] > 1954.8) & (_input0['YearBuilt'] <= 1968.6), 'YearBuilt'] = 5
_input0.loc[(_input0['YearBuilt'] > 1968.6) & (_input0['YearBuilt'] <= 1982.4), 'YearBuilt'] = 6
_input0.loc[(_input0['YearBuilt'] > 1982.4) & (_input0['YearBuilt'] <= 1996.2), 'YearBuilt'] = 7
_input0.loc[_input0['YearBuilt'] > 1996.2, 'YearBuilt'] = 8
_input0['YearBuilt'].value_counts()
_input1['YearRemodAdd_range'] = pd.cut(_input1['YearRemodAdd'], 10)
_input1['YearRemodAdd_range'].value_counts()
_input0['YearRemodAdd_range'] = pd.cut(_input0['YearRemodAdd'], 10)
_input0['YearRemodAdd_range'].value_counts()
_input1.loc[_input1['YearRemodAdd'] <= 1956, 'YearRemodAdd'] = 0
_input1.loc[(_input1['YearRemodAdd'] > 1956) & (_input1['YearRemodAdd'] <= 1962), 'YearRemodAdd'] = 1
_input1.loc[(_input1['YearRemodAdd'] > 1962) & (_input1['YearRemodAdd'] <= 1968), 'YearRemodAdd'] = 2
_input1.loc[(_input1['YearRemodAdd'] > 1968) & (_input1['YearRemodAdd'] <= 1974), 'YearRemodAdd'] = 3
_input1.loc[(_input1['YearRemodAdd'] > 1974) & (_input1['YearRemodAdd'] <= 1980), 'YearRemodAdd'] = 4
_input1.loc[(_input1['YearRemodAdd'] > 1980) & (_input1['YearRemodAdd'] <= 1986), 'YearRemodAdd'] = 5
_input1.loc[(_input1['YearRemodAdd'] > 1986) & (_input1['YearRemodAdd'] <= 1992), 'YearRemodAdd'] = 6
_input1.loc[(_input1['YearRemodAdd'] > 1992) & (_input1['YearRemodAdd'] <= 1998), 'YearRemodAdd'] = 7
_input1.loc[_input1['YearRemodAdd'] > 1998, 'YearRemodAdd'] = 8
_input0.loc[_input0['YearRemodAdd'] <= 1956, 'YearRemodAdd'] = 0
_input0.loc[(_input0['YearRemodAdd'] > 1956) & (_input0['YearRemodAdd'] <= 1962), 'YearRemodAdd'] = 1
_input0.loc[(_input0['YearRemodAdd'] > 1962) & (_input0['YearRemodAdd'] <= 1968), 'YearRemodAdd'] = 2
_input0.loc[(_input0['YearRemodAdd'] > 1968) & (_input0['YearRemodAdd'] <= 1974), 'YearRemodAdd'] = 3
_input0.loc[(_input0['YearRemodAdd'] > 1974) & (_input0['YearRemodAdd'] <= 1980), 'YearRemodAdd'] = 4
_input0.loc[(_input0['YearRemodAdd'] > 1980) & (_input0['YearRemodAdd'] <= 1986), 'YearRemodAdd'] = 5
_input0.loc[(_input0['YearRemodAdd'] > 1986) & (_input0['YearRemodAdd'] <= 1992), 'YearRemodAdd'] = 6
_input0.loc[(_input0['YearRemodAdd'] > 1992) & (_input0['YearRemodAdd'] <= 1998), 'YearRemodAdd'] = 7
_input0.loc[_input0['YearRemodAdd'] > 1998, 'YearRemodAdd'] = 8
_input1 = _input1.drop(['Year_built_range', 'YearRemodAdd_range'], axis=1)
_input0 = _input0.drop('YearRemodAdd_range', axis=1)
yrsold_mapping = {2006: 0, 2007: 1, 2008: 2, 2009: 3, 2010: 4}
_input1['YrSold'] = _input1['YrSold'].map(yrsold_mapping)
_input0['YrSold'] = _input0['YrSold'].map(yrsold_mapping)
_input0['YrSold'].value_counts()
_input1['LotFrontage_range'] = pd.cut(_input1['LotFrontage'], 10)
_input1['LotFrontage_range'].value_counts()
_input1.loc[_input1['LotFrontage'] <= 50.2, 'LotFrontage'] = 0
_input1.loc[(_input1['LotFrontage'] > 50.2) & (_input1['LotFrontage'] <= 79.4), 'LotFrontage'] = 1
_input1.loc[(_input1['LotFrontage'] > 79.4) & (_input1['LotFrontage'] <= 108.6), 'LotFrontage'] = 2
_input1.loc[(_input1['LotFrontage'] > 108.6) & (_input1['LotFrontage'] <= 137.8), 'LotFrontage'] = 3
_input1.loc[(_input1['LotFrontage'] > 137.8) & (_input1['LotFrontage'] <= 167), 'LotFrontage'] = 4
_input1.loc[(_input1['LotFrontage'] > 167) & (_input1['LotFrontage'] <= 196.2), 'LotFrontage'] = 5
_input1.loc[(_input1['LotFrontage'] > 196.2) & (_input1['LotFrontage'] <= 225.4), 'LotFrontage'] = 6
_input1.loc[(_input1['LotFrontage'] > 225.4) & (_input1['LotFrontage'] <= 254.6), 'LotFrontage'] = 7
_input1.loc[_input1['LotFrontage'] > 254.6, 'LotFrontage'] = 8
_input0.loc[_input0['LotFrontage'] <= 50.2, 'LotFrontage'] = 0
_input0.loc[(_input0['LotFrontage'] > 50.2) & (_input0['LotFrontage'] <= 79.4), 'LotFrontage'] = 1
_input0.loc[(_input0['LotFrontage'] > 79.4) & (_input0['LotFrontage'] <= 108.6), 'LotFrontage'] = 2
_input0.loc[(_input0['LotFrontage'] > 108.6) & (_input0['LotFrontage'] <= 137.8), 'LotFrontage'] = 3
_input0.loc[(_input0['LotFrontage'] > 137.8) & (_input0['LotFrontage'] <= 167), 'LotFrontage'] = 4
_input0.loc[(_input0['LotFrontage'] > 167) & (_input0['LotFrontage'] <= 196.2), 'LotFrontage'] = 5
_input0.loc[(_input0['LotFrontage'] > 196.2) & (_input0['LotFrontage'] <= 225.4), 'LotFrontage'] = 6
_input0.loc[(_input0['LotFrontage'] > 225.4) & (_input0['LotFrontage'] <= 254.6), 'LotFrontage'] = 7
_input0.loc[_input0['LotFrontage'] > 254.6, 'LotFrontage'] = 8
print(_input1['LotFrontage'].value_counts())
print(_input0['LotFrontage'].value_counts())
_input1 = _input1.drop('LotFrontage_range', axis=1)
_input1['TotalBsmtSF_range'] = pd.cut(_input1['TotalBsmtSF'], 10)
_input1['TotalBsmtSF_range'].value_counts()
_input1.loc[_input1['TotalBsmtSF'] <= 611, 'TotalBsmtSF'] = 0
_input1.loc[(_input1['TotalBsmtSF'] > 611) & (_input1['TotalBsmtSF'] <= 1222), 'TotalBsmtSF'] = 1
_input1.loc[(_input1['TotalBsmtSF'] > 1222) & (_input1['TotalBsmtSF'] <= 1833), 'TotalBsmtSF'] = 2
_input1.loc[(_input1['TotalBsmtSF'] > 1833) & (_input1['TotalBsmtSF'] <= 2444), 'TotalBsmtSF'] = 3
_input1.loc[(_input1['TotalBsmtSF'] > 2444) & (_input1['TotalBsmtSF'] <= 3055), 'TotalBsmtSF'] = 4
_input1.loc[(_input1['TotalBsmtSF'] > 3055) & (_input1['TotalBsmtSF'] <= 3666), 'TotalBsmtSF'] = 5
_input1.loc[(_input1['TotalBsmtSF'] > 3666) & (_input1['TotalBsmtSF'] <= 4277), 'TotalBsmtSF'] = 6
_input1.loc[(_input1['TotalBsmtSF'] > 4277) & (_input1['TotalBsmtSF'] <= 4888), 'TotalBsmtSF'] = 7
_input1.loc[_input1['TotalBsmtSF'] > 4888, 'TotalBsmtSF'] = 8
_input0.loc[_input0['TotalBsmtSF'] <= 611, 'TotalBsmtSF'] = 0
_input0.loc[(_input0['TotalBsmtSF'] > 611) & (_input0['TotalBsmtSF'] <= 1222), 'TotalBsmtSF'] = 1
_input0.loc[(_input0['TotalBsmtSF'] > 1222) & (_input0['TotalBsmtSF'] <= 1833), 'TotalBsmtSF'] = 2
_input0.loc[(_input0['TotalBsmtSF'] > 1833) & (_input0['TotalBsmtSF'] <= 2444), 'TotalBsmtSF'] = 3
_input0.loc[(_input0['TotalBsmtSF'] > 2444) & (_input0['TotalBsmtSF'] <= 3055), 'TotalBsmtSF'] = 4
_input0.loc[(_input0['TotalBsmtSF'] > 3055) & (_input0['TotalBsmtSF'] <= 3666), 'TotalBsmtSF'] = 5
_input0.loc[(_input0['TotalBsmtSF'] > 3666) & (_input0['TotalBsmtSF'] <= 4277), 'TotalBsmtSF'] = 6
_input0.loc[(_input0['TotalBsmtSF'] > 4277) & (_input0['TotalBsmtSF'] <= 4888), 'TotalBsmtSF'] = 7
_input0.loc[_input0['TotalBsmtSF'] > 4888, 'TotalBsmtSF'] = 8
_input1 = _input1.drop('TotalBsmtSF_range', axis=1)
_input1['GrLivArea_range'] = pd.cut(_input1['GrLivArea'], 10)
_input1.loc[_input1['GrLivArea'] <= 864.8, 'GrLivArea'] = 0
_input1.loc[(_input1['GrLivArea'] > 864.8) & (_input1['GrLivArea'] <= 1395.6), 'GrLivArea'] = 1
_input1.loc[(_input1['GrLivArea'] > 1395.6) & (_input1['GrLivArea'] <= 1926.4), 'GrLivArea'] = 2
_input1.loc[(_input1['GrLivArea'] > 1926.4) & (_input1['GrLivArea'] <= 2457.2), 'GrLivArea'] = 3
_input1.loc[(_input1['GrLivArea'] > 2457.2) & (_input1['GrLivArea'] <= 2988), 'GrLivArea'] = 4
_input1.loc[(_input1['GrLivArea'] > 2988) & (_input1['GrLivArea'] <= 3518.8), 'GrLivArea'] = 5
_input1.loc[(_input1['GrLivArea'] > 3518.8) & (_input1['GrLivArea'] <= 4049.6), 'GrLivArea'] = 6
_input1.loc[(_input1['GrLivArea'] > 4049.6) & (_input1['GrLivArea'] <= 4580.4), 'GrLivArea'] = 7
_input1.loc[_input1['GrLivArea'] > 4580.4, 'GrLivArea'] = 8
_input0.loc[_input0['GrLivArea'] <= 864.8, 'GrLivArea'] = 0
_input0.loc[(_input0['GrLivArea'] > 864.8) & (_input0['GrLivArea'] <= 1395.6), 'GrLivArea'] = 1
_input0.loc[(_input0['GrLivArea'] > 1395.6) & (_input0['GrLivArea'] <= 1926.4), 'GrLivArea'] = 2
_input0.loc[(_input0['GrLivArea'] > 1926.4) & (_input0['GrLivArea'] <= 2457.2), 'GrLivArea'] = 3
_input0.loc[(_input0['GrLivArea'] > 2457.2) & (_input0['GrLivArea'] <= 2988), 'GrLivArea'] = 4
_input0.loc[(_input0['GrLivArea'] > 2988) & (_input0['GrLivArea'] <= 3518.8), 'GrLivArea'] = 5
_input0.loc[(_input0['GrLivArea'] > 3518.8) & (_input0['GrLivArea'] <= 4049.6), 'GrLivArea'] = 6
_input0.loc[(_input0['GrLivArea'] > 4049.6) & (_input0['GrLivArea'] <= 4580.4), 'GrLivArea'] = 7
_input0.loc[_input0['GrLivArea'] > 4580.4, 'GrLivArea'] = 8
print(_input1['GrLivArea'].value_counts())
print(_input0['GrLivArea'].value_counts())
_input1 = _input1.drop('GrLivArea_range', axis=1)
_input1['WoodDeckSF_range'] = pd.cut(_input1['WoodDeckSF'], 10)
_input1['WoodDeckSF_range'].value_counts()
_input1.loc[_input1['WoodDeckSF'] <= 85.7, 'WoodDeckSF'] = 0
_input1.loc[(_input1['WoodDeckSF'] > 85.7) & (_input1['WoodDeckSF'] <= 171.4), 'WoodDeckSF'] = 1
_input1.loc[(_input1['WoodDeckSF'] > 171.4) & (_input1['WoodDeckSF'] <= 257.1), 'WoodDeckSF'] = 2
_input1.loc[(_input1['WoodDeckSF'] > 257.1) & (_input1['WoodDeckSF'] <= 342.8), 'WoodDeckSF'] = 3
_input1.loc[(_input1['WoodDeckSF'] > 342.8) & (_input1['WoodDeckSF'] <= 428.5), 'WoodDeckSF'] = 4
_input1.loc[(_input1['WoodDeckSF'] > 428.5) & (_input1['WoodDeckSF'] <= 514.2), 'WoodDeckSF'] = 5
_input1.loc[(_input1['WoodDeckSF'] > 514.2) & (_input1['WoodDeckSF'] <= 599.9), 'WoodDeckSF'] = 6
_input1.loc[(_input1['WoodDeckSF'] > 599.9) & (_input1['WoodDeckSF'] <= 685.6), 'WoodDeckSF'] = 7
_input1.loc[_input1['WoodDeckSF'] > 685.6, 'WoodDeckSF'] = 8
_input0.loc[_input0['WoodDeckSF'] <= 85.7, 'WoodDeckSF'] = 0
_input0.loc[(_input0['WoodDeckSF'] > 85.7) & (_input0['WoodDeckSF'] <= 171.4), 'WoodDeckSF'] = 1
_input0.loc[(_input0['WoodDeckSF'] > 171.4) & (_input0['WoodDeckSF'] <= 257.1), 'WoodDeckSF'] = 2
_input0.loc[(_input0['WoodDeckSF'] > 257.1) & (_input0['WoodDeckSF'] <= 342.8), 'WoodDeckSF'] = 3
_input0.loc[(_input0['WoodDeckSF'] > 342.8) & (_input0['WoodDeckSF'] <= 428.5), 'WoodDeckSF'] = 4
_input0.loc[(_input0['WoodDeckSF'] > 428.5) & (_input0['WoodDeckSF'] <= 514.2), 'WoodDeckSF'] = 5
_input0.loc[(_input0['WoodDeckSF'] > 514.2) & (_input0['WoodDeckSF'] <= 599.9), 'WoodDeckSF'] = 6
_input0.loc[(_input0['WoodDeckSF'] > 599.9) & (_input0['WoodDeckSF'] <= 685.6), 'WoodDeckSF'] = 7
_input0.loc[_input0['WoodDeckSF'] > 685.6, 'WoodDeckSF'] = 8
_input0['WoodDeckSF'].value_counts()
_input1 = _input1.drop('WoodDeckSF_range', axis=1)
_input1['OpenPorchSF'].value_counts()
_input1 = _input1.drop('3SsnPorch', axis=1)
_input0 = _input0.drop('3SsnPorch', axis=1)
_input1 = _input1.drop('ScreenPorch', axis=1)
_input0 = _input0.drop('ScreenPorch', axis=1)
_input1['OpenPorchSF_range'] = pd.cut(_input1['OpenPorchSF'], 10)
_input1['OpenPorchSF_range'].value_counts()
_input1.loc[_input1['OpenPorchSF'] <= 54.7, 'OpenPorchSF'] = 0
_input1.loc[(_input1['OpenPorchSF'] > 54.7) & (_input1['OpenPorchSF'] <= 109.4), 'OpenPorchSF'] = 1
_input1.loc[(_input1['OpenPorchSF'] > 109.4) & (_input1['OpenPorchSF'] <= 164.1), 'OpenPorchSF'] = 2
_input1.loc[(_input1['OpenPorchSF'] > 164.1) & (_input1['OpenPorchSF'] <= 218.8), 'OpenPorchSF'] = 3
_input1.loc[(_input1['OpenPorchSF'] > 218.8) & (_input1['OpenPorchSF'] <= 273.5), 'OpenPorchSF'] = 4
_input1.loc[(_input1['OpenPorchSF'] > 273.5) & (_input1['OpenPorchSF'] <= 328.2), 'OpenPorchSF'] = 5
_input1.loc[(_input1['OpenPorchSF'] > 328.2) & (_input1['OpenPorchSF'] <= 382.9), 'OpenPorchSF'] = 6
_input1.loc[(_input1['OpenPorchSF'] > 382.9) & (_input1['OpenPorchSF'] <= 437.6), 'OpenPorchSF'] = 7
_input1.loc[_input1['OpenPorchSF'] > 437.6, 'OpenPorchSF'] = 8
_input0.loc[_input0['OpenPorchSF'] <= 54.7, 'OpenPorchSF'] = 0
_input0.loc[(_input0['OpenPorchSF'] > 54.7) & (_input0['OpenPorchSF'] <= 109.4), 'OpenPorchSF'] = 1
_input0.loc[(_input0['OpenPorchSF'] > 109.4) & (_input0['OpenPorchSF'] <= 164.1), 'OpenPorchSF'] = 2
_input0.loc[(_input0['OpenPorchSF'] > 164.1) & (_input0['OpenPorchSF'] <= 218.8), 'OpenPorchSF'] = 3
_input0.loc[(_input0['OpenPorchSF'] > 218.8) & (_input0['OpenPorchSF'] <= 273.5), 'OpenPorchSF'] = 4
_input0.loc[(_input0['OpenPorchSF'] > 273.5) & (_input0['OpenPorchSF'] <= 328.2), 'OpenPorchSF'] = 5
_input0.loc[(_input0['OpenPorchSF'] > 328.2) & (_input0['OpenPorchSF'] <= 382.9), 'OpenPorchSF'] = 6
_input0.loc[(_input0['OpenPorchSF'] > 382.9) & (_input0['OpenPorchSF'] <= 437.6), 'OpenPorchSF'] = 7
_input0.loc[_input0['OpenPorchSF'] > 437.6, 'OpenPorchSF'] = 8
_input1.head()
_input1 = _input1.drop('OpenPorchSF_range', axis=1)
_input0.head()
for (counts, column) in enumerate(_input1.columns):
    print(counts, column + ':', _input1[column].isnull().sum(), _input1[column].dtype)
for (counts, column) in enumerate(_input0.columns):
    print(counts, column + ':', _input0[column].isnull().sum(), _input0[column].dtype)
_input1.head()
X_train = _input1.drop(['Id', 'SalePrice', 'LogSalePrice'], axis=1)
y_train = _input1['LogSalePrice']
X_test = _input0.drop('Id', axis=1)
lr = LinearRegression()