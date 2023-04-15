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
train_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
sample_submission_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
train_data[train_data['BsmtFinSF1'] == 0][['BsmtFinSF1', 'BsmtFinSF2']]
for i in [train_data, test_data]:
    i.info()
train_data[['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']]
print('The length of Test Data given in csv file is ', len(test_data.columns), '\nThe length of Train Data in csv file is ', len(train_data.columns))
train_data.describe()
test_data.describe()
train_data['MoSold'].value_counts()
test_data['MoSold'].value_counts()
plt.figure(figsize=(14, 8))
sns.heatmap(train_data.isnull(), yticklabels=False, cmap='viridis')

plt.figure(figsize=(14, 8))
sns.heatmap(test_data.isnull(), yticklabels=False, cmap='viridis')

na_data = train_data.isnull().sum() / len(train_data) * 100
na_data = na_data.drop(na_data[na_data == 0].index).sort_values(ascending=False)[:15]
missing_data_analysis = pd.DataFrame({'Missing Data': na_data})
missing_data_analysis
all_testdata_na = test_data.isnull().sum() / len(test_data) * 100
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

for column in train_data.columns:
    print(column + ':', train_data[column].isnull().sum(), train_data[column].dtype)
train_data.isnull().sum().idxmax()
train_data[train_data['GarageYrBlt'].isnull() & (train_data['YearBuilt'] > 1800)]
for column in test_data.columns:
    print(column + ':', test_data[column].isnull().sum(), 'dtype: ', test_data[column].dtype)
test_data.isnull().sum().idxmax()
train_data['PoolQC'].isnull().sum()
plt.figure(figsize=(14, 8))
sns.heatmap(train_data.corr())

plt.figure(figsize=(14, 8))
train_data.corr()['SalePrice'].sort_values()[:-1].plot(kind='bar')

train_data.corr()['SalePrice'].sort_values(ascending=False)[1:]
train_data['SalePrice'].sort_values(ascending=True)[:10]
plt.figure(figsize=(14, 8))
sns.distplot(train_data['SalePrice'], kde=True, bins=50)

train_data[train_data['SalePrice'] > 600000]
train_data[train_data['SalePrice'] < 50000]
plt.figure(figsize=(14, 8))
sns.distplot(train_data['SalePrice'], kde=True, bins=50)

(fig, axes) = plt.subplots(1, 2, figsize=(18, 8))
sns.scatterplot(x='SalePrice', y='YearBuilt', data=train_data, ax=axes[0])
sns.scatterplot(x='SalePrice', y='YearRemodAdd', data=train_data, ax=axes[1])
axes[0].set_title('SalePrice vs YearBuilt')
axes[1].set_title('SalePrice vs Year Remodeled')

plt.figure(figsize=(14, 8))
stats.probplot(train_data['SalePrice'], plot=plt, dist='norm')

train_data['LogSalePrice'] = np.log(train_data['SalePrice'])
plt.figure(figsize=(14, 8))
sns.distplot(train_data['LogSalePrice'], kde=True, bins=50)

plt.figure(figsize=(14, 8))
stats.probplot(train_data['LogSalePrice'], plot=plt, dist='norm')

sns.boxplot(data=train_data, y='LogSalePrice')
train_data['LogSalePrice'].describe()
(fig, ax) = plt.subplots(1, 2, figsize=(14, 8))
sns.countplot(data=train_data, x='GarageCars', ax=ax[0])
sns.countplot(data=test_data, x='GarageCars', ax=ax[1])
ax[0].set_title('Number Car Garage Training Set')
ax[1].set_title('Number Car Garage Test Set')

print(train_data['GarageCars'].value_counts())
print(test_data['GarageCars'].value_counts())
print(train_data['GarageCars'].isnull().sum())
print(test_data['GarageCars'].isnull().sum())
print(train_data['GrLivArea'].value_counts())
print(test_data['GrLivArea'].value_counts())
print(train_data['GrLivArea'].isnull().sum())
print(test_data['GrLivArea'].isnull().sum())
(fig, ax) = plt.subplots(1, 2, figsize=(14, 8))
sns.histplot(x=train_data['GrLivArea'], ax=ax[0], bins=50)
sns.histplot(x=test_data['GrLivArea'], ax=ax[1], bins=50)
ax[0].set_title('Above Ground Living Area Training Set')
ax[1].set_title('Above Ground Living Area Testing Set')

print('mode: ', train_data['SalePrice'].mode())
print('median: ', train_data['SalePrice'].median())
print('mean: ', train_data['SalePrice'].mean())
print('min: ', train_data['SalePrice'].min())
print('max: ', train_data['SalePrice'].max())
print(train_data['SalePrice'].count())
for i in train_data.columns:
    print(i + ': ', train_data[i].dtypes)
print('train_data\n', train_data.dtypes.value_counts())
print('test_data\n', test_data.dtypes.value_counts())
print(train_data['Street'].value_counts())
print(train_data['Street'].isnull().sum())
print(train_data['Alley'].value_counts())
print(train_data['Alley'].isnull().sum())
print(test_data['Alley'].value_counts())
print(test_data['Alley'].isnull().sum())
for column in train_data.columns:
    if train_data[column].isnull().sum() / len(train_data) > 0.5:
        print(column)
train_data = train_data.drop(['Alley', 'Fence', 'FireplaceQu', 'MiscFeature'], axis=1)
for column in test_data.columns:
    if test_data[column].isnull().sum() / len(test_data) > 0.5:
        print(column)
test_data = test_data.drop(['FireplaceQu', 'Alley', 'Fence', 'MiscFeature'], axis=1)
train_data['LotFrontage'] = train_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
test_data['LotFrontage'] = test_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
plt.figure(figsize=(14, 8))
sns.heatmap(train_data.isnull(), yticklabels=False, cmap='viridis')

plt.figure(figsize=(14, 8))
sns.heatmap(test_data.isnull(), yticklabels=False, cmap='viridis')

plt.figure(figsize=(20, 12))
sns.heatmap(train_data.corr(), annot=False)

plt.figure(figsize=(14, 8))
sns.countplot(data=train_data, x=train_data['Heating'])

train_data['Heating'].value_counts()
(fig, ax) = plt.subplots(1, 2, figsize=(14, 8))
sns.countplot(data=train_data, x='GarageCars', hue='PavedDrive', ax=ax[0])
sns.countplot(data=train_data, x='GarageQual', ax=ax[1])
ax[0].set_title('#Cars Garage vs. Paved/Not')
ax[1].set_title('Garage Quality')

(fig, ax) = plt.subplots(1, 2, figsize=(14, 8))
sns.countplot(data=test_data, x='GarageCars', hue='PavedDrive', ax=ax[0])
sns.countplot(data=test_data, x='GarageQual', ax=ax[1])
ax[0].set_title('#Cars Garage vs. Paved/Not...Test set')
ax[1].set_title('Garage Quality...Test set')

plt.figure(figsize=(14, 8))
sns.countplot(data=test_data, x='MSZoning', hue='Neighborhood')
plt.title('MSZoning vs Neighborhood Test set')
plt.legend(loc=1)

plt.figure(figsize=(14, 8))
train_data.corr()['SalePrice'].sort_values()[22:-1].plot(kind='bar')

(fig, ax) = plt.subplots(1, 2, figsize=(14, 8))
sns.countplot(data=train_data, x='Fireplaces', ax=ax[0])
sns.countplot(data=test_data, x='Fireplaces', ax=ax[1])
ax[0].set_title('# Fireplaces Training set')
ax[1].set_title('# Fireplaces Test set')

print(train_data['GarageQual'].isnull().sum())
print(test_data['GarageQual'].isnull().sum())
print('OverallQual_Train', train_data['OverallQual'].isnull().sum())
print('OverallQual_Test', test_data['OverallQual'].isnull().sum())
print('GarageQual_Train', train_data['GarageQual'].isnull().sum())
print('GarageQual_Test', test_data['GarageQual'].isnull().sum())
print('GrLivArea_Train', train_data['GrLivArea'].isnull().sum())
print('GrLivArea_Test', test_data['GrLivArea'].isnull().sum())
print('GarageCars_Train', train_data['GarageCars'].isnull().sum())
print('GarageCars_Test', test_data['GarageCars'].isnull().sum())
print('TotalBsmtSF_Train', train_data['TotalBsmtSF'].isnull().sum())
print('TotalBmstSF_Test', test_data['TotalBsmtSF'].isnull().sum())
print('1stFlrSF_Train', train_data['1stFlrSF'].isnull().sum())
print('1stFlrSF_test', test_data['1stFlrSF'].isnull().sum())
print('FullBath', train_data['FullBath'].isnull().sum())
print('FullBath', test_data['FullBath'].isnull().sum())
print('TotalRmsAbvGrd_Train', train_data['TotRmsAbvGrd'].isnull().sum())
print('TotalRmsAbvGrd_Test', test_data['TotRmsAbvGrd'].isnull().sum())
print('YearBuilt_Train', train_data['YearBuilt'].isnull().sum())
print('YearBuilt_Test', test_data['YearBuilt'].isnull().sum())
print('YearRemodeled_Train', train_data['YearRemodAdd'].isnull().sum())
print('YearRemodeled_Test', test_data['YearRemodAdd'].isnull().sum())
train_data['GarageQual'] = train_data['GarageQual'].fillna('None')
test_data['GarageQual'] = test_data['GarageQual'].fillna('None')
test_data['MSZoning'] = test_data.groupby('Neighborhood')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
test_data[test_data['TotalBsmtSF'].isnull()]
test_data['GarageType'].isnull().sum()
test_data[test_data['GarageType'].isnull()]
train_data['BsmtQual'].isnull().sum()
df = train_data[train_data['BsmtQual'].isnull()]
df[df.columns[30:35]].head()
train_data['BsmtCond'] = train_data['BsmtCond'].fillna('None')
train_data['BsmtExposure'] = train_data['BsmtExposure'].fillna('None')
train_data['BsmtFinType1'] = train_data['BsmtFinType1'].fillna('None')
train_data['BsmtFinType2'] = train_data['BsmtFinType2'].fillna('None')
print(train_data['BsmtCond'].isnull().sum())
print(train_data['BsmtExposure'].isnull().sum())
print(train_data['BsmtFinType1'].isnull().sum())
print(train_data['BsmtFinType2'].isnull().sum())
test_data['BsmtCond'] = test_data['BsmtCond'].fillna('None')
test_data['BsmtExposure'] = test_data['BsmtExposure'].fillna('None')
test_data['BsmtFinType1'] = test_data['BsmtFinType1'].fillna('None')
test_data['BsmtFinType2'] = test_data['BsmtFinType2'].fillna('None')
print(test_data['BsmtCond'].isnull().sum())
print(test_data['BsmtExposure'].isnull().sum())
print(test_data['BsmtFinType1'].isnull().sum())
print(test_data['BsmtFinType2'].isnull().sum())
plt.figure(figsize=(14, 8))
sns.heatmap(test_data.isnull(), yticklabels=False, cmap='viridis')

plt.figure(figsize=(14, 8))
sns.heatmap(train_data.isnull(), yticklabels=False, cmap='viridis')

train_data['MasVnrType'].isnull().sum()
train_data[train_data['MasVnrType'].isnull()]
plt.figure(figsize=(14, 8))
sns.countplot(data=train_data, x='Exterior1st', hue='MasVnrType')
plt.legend(loc=1)

plt.figure(figsize=(14, 8))
sns.countplot(data=train_data, x='Exterior2nd', hue='MasVnrType')
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
train_data['MasVnrType'] = train_data[['MasVnrType', 'Exterior1st']].apply(mason_veneer, axis=1)
train_data[train_data['MasVnrType'] == 'None'][['MasVnrType', 'MasVnrArea']]
plt.figure(figsize=(14, 8))
sns.heatmap(train_data.isnull(), yticklabels=False, cmap='viridis')
plt.figure(figsize=(14, 8))
sns.heatmap(test_data.isnull(), yticklabels=False, cmap='viridis')
train_data['MasVnrArea'] = train_data['MasVnrArea'].fillna(0)
test_data['MasVnrArea'] = test_data['MasVnrArea'].fillna(0)
plt.figure(figsize=(14, 8))
sns.heatmap(train_data.isnull(), yticklabels=False, cmap='viridis')
plt.figure(figsize=(14, 8))
sns.heatmap(test_data.isnull(), yticklabels=False, cmap='viridis')
for column in test_data.columns:
    print(column + ':', test_data[column].isnull().sum(), test_data[column].dtype)
train_data = train_data.drop(['PoolArea', 'PoolQC'], axis=1)
test_data = test_data.drop(['PoolArea', 'PoolQC'], axis=1)
train_data = train_data.drop(['MSSubClass', 'MSZoning', 'LotArea', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope'], axis=1)
test_data = test_data.drop(['MSSubClass', 'MSZoning', 'LotArea', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope'], axis=1)
train_data = train_data.drop('Condition2', axis=1)
test_data = test_data.drop('Condition2', axis=1)
train_data = train_data.drop(['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF'], axis=1)
test_data = test_data.drop(['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF'], axis=1)
train_data = train_data.drop(['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageArea', 'GarageQual', 'GarageCond'], axis=1)
test_data = test_data.drop(['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageArea', 'GarageQual', 'GarageCond'], axis=1)
train_data = train_data.drop(['1stFlrSF', '2ndFlrSF', 'LowQualFinSF'], axis=1)
test_data = test_data.drop(['1stFlrSF', '2ndFlrSF', 'LowQualFinSF'], axis=1)
train_data = train_data.drop(['RoofStyle', 'RoofMatl', 'BsmtFinType2'], axis=1)
test_data = test_data.drop(['RoofStyle', 'RoofMatl', 'BsmtFinType2'], axis=1)
train_data = train_data.drop(['Exterior2nd', 'BsmtExposure', 'Electrical', 'MiscVal'], axis=1)
test_data = test_data.drop(['Exterior2nd', 'BsmtExposure', 'Electrical', 'MiscVal'], axis=1)
train_data = train_data.drop('Functional', axis=1)
test_data = test_data.drop('Functional', axis=1)
train_data = train_data.drop(['MasVnrType', 'MasVnrArea'], axis=1)
test_data = test_data.drop(['MasVnrType', 'MasVnrArea'], axis=1)
train_data = train_data.drop('BsmtFinType1', axis=1)
test_data = test_data.drop('BsmtFinType1', axis=1)
train_data = train_data.drop('SaleType', axis=1)
test_data = test_data.drop('SaleType', axis=1)
train_data['TotHalfBaths'] = train_data['BsmtHalfBath'] + train_data['HalfBath']
train_data['TotalFullBaths'] = train_data['BsmtFullBath'] + train_data['FullBath']
test_data['TotHalfBaths'] = test_data['BsmtHalfBath'] + test_data['HalfBath']
test_data['TotalFullBaths'] = test_data['BsmtFullBath'] + test_data['FullBath']
train_data = train_data.drop(['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath'], axis=1)
test_data = test_data.drop(['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath'], axis=1)
test_data['TotalBsmtSF'] = test_data['TotalBsmtSF'].fillna(round(test_data['TotalBsmtSF'].mean(), 0))
test_data['GarageCars'] = test_data['GarageCars'].fillna(0)
test_data['Exterior1st'] = test_data['Exterior1st'].fillna(test_data['Exterior1st'].mode()[0])
test_data['BsmtQual'] = test_data['BsmtQual'].fillna('None')
train_data['BsmtQual'] = train_data['BsmtQual'].fillna('None')
test_data['KitchenQual'] = test_data['KitchenQual'].fillna(test_data['KitchenQual'].mode()[0])
train_data.groupby('Neighborhood')['SalePrice'].mean().sort_values(ascending=False)
plt.figure(figsize=(14, 8))
sns.heatmap(train_data.isnull(), yticklabels=False, cmap='viridis')

plt.figure(figsize=(14, 8))
sns.heatmap(test_data.isnull(), yticklabels=False, cmap='viridis')

train_data['TotHalfBaths'] = train_data['TotHalfBaths'].fillna(train_data['TotHalfBaths'].mode()[0])
test_data['TotHalfBaths'] = test_data['TotHalfBaths'].fillna(test_data['TotHalfBaths'].mode()[0])
train_data['TotalFullBaths'] = train_data['TotalFullBaths'].fillna(train_data['TotalFullBaths'].mode()[0])
test_data['TotalFullBaths'] = test_data['TotalFullBaths'].fillna(test_data['TotalFullBaths'].mode()[0])
train_data['HeatingQC'].value_counts()
heatingqc_mapping = {'Ex': 0, 'TA': 1, 'Gd': 2, 'Fa': 3, 'Po': 4}
train_data['HeatingQC'] = train_data['HeatingQC'].map(heatingqc_mapping)
test_data['HeatingQC'] = test_data['HeatingQC'].map(heatingqc_mapping)
train_data.head()
print(train_data['HouseStyle'].value_counts())
print(test_data['HouseStyle'].value_counts())
housestyle_mapping = {'1Story': 0, '2Story': 1, '1.5Fin': 2, 'SLvl': 3, 'SFoyer': 4, '1.5Unf': 5, '2.5Unf': 6, '2.5Fin': 7}
train_data['HouseStyle'] = train_data['HouseStyle'].map(housestyle_mapping)
test_data['HouseStyle'] = test_data['HouseStyle'].map(housestyle_mapping)
print(train_data['BsmtCond'].value_counts())
print(test_data['BsmtCond'].value_counts())
bsmtcond_mapping = {'None': 0, 'TA': 1, 'Gd': 2, 'Fa': 3, 'Po': 4}
train_data['BsmtCond'] = train_data['BsmtCond'].map(bsmtcond_mapping)
test_data['BsmtCond'] = test_data['BsmtCond'].map(bsmtcond_mapping)
train_data['ExterCond'].value_counts()
extercond_mapping = {'TA': 0, 'Gd': 1, 'Fa': 2, 'Ex': 3, 'Po': 4}
train_data['ExterCond'] = train_data['ExterCond'].map(extercond_mapping)
test_data['ExterCond'] = test_data['ExterCond'].map(extercond_mapping)
print(train_data['ExterCond'].value_counts())
print(test_data['ExterCond'].value_counts())
neighborhood_mapping = {'NridgHt': 0, 'StoneBr': 1, 'NoRidge': 2, 'Timber': 3, 'Veenker': 4, 'Somerst': 5, 'ClearCr': 6, 'Crawfor': 7, 'CollgCr': 8, 'Blmngtn': 9, 'Gilbert': 10, 'NWAmes': 11, 'SawyerW': 12, 'Mitchel': 13, 'NAmes': 14, 'NPkVill': 15, 'SWISU': 16, 'Blueste': 17, 'Sawyer': 18, 'OldTown': 19, 'Edwards': 20, 'BrkSide': 21, 'BrDale': 22, 'IDOTRR': 23, 'MeadowV': 24}
train_data['Neighborhood'] = train_data['Neighborhood'].map(neighborhood_mapping)
test_data['Neighborhood'] = test_data['Neighborhood'].map(neighborhood_mapping)
ac_mapping = {'N': 0, 'Y': 1}
train_data['CentralAir'] = train_data['CentralAir'].map(ac_mapping)
test_data['CentralAir'] = test_data['CentralAir'].map(ac_mapping)
heat_mapping = {'Floor': 0, 'OthW': 1, 'Wall': 2, 'Grav': 3, 'GasW': 4, 'GasA': 5}
train_data['Heating'] = train_data['Heating'].map(heat_mapping)
test_data['Heating'] = test_data['Heating'].map(heat_mapping)
condition_mapping = {'Norm': 0, 'Feedr': 1, 'Artery': 2, 'RRAn': 3, 'PosN': 4, 'RRAe': 5, 'PosA': 6, 'RRNn': 7, 'RRNe': 8}
train_data['Condition1'] = train_data['Condition1'].map(condition_mapping)
test_data['Condition1'] = test_data['Condition1'].map(condition_mapping)
bldg_mapping = {'1Fam': 0, 'TwnhsE': 1, 'Duplex': 2, 'Twnhs': 3, '2fmCon': 4}
train_data['BldgType'] = train_data['BldgType'].map(bldg_mapping)
test_data['BldgType'] = test_data['BldgType'].map(bldg_mapping)
sale_condition_mapping = {'Normal': 0, 'Partial': 1, 'Abnorml': 2, 'Family': 3, 'Alloca': 4, 'AdjLand': 5}
train_data['SaleCondition'] = train_data['SaleCondition'].map(sale_condition_mapping)
test_data['SaleCondition'] = test_data['SaleCondition'].map(sale_condition_mapping)
train_data['Exterior1st'].value_counts()
train_data['Exterior1st'] = train_data['Exterior1st'].replace(['WdShing', 'Stucco', 'AsbShng', 'BrkComm', 'Stone', 'AsphShn', 'ImStucc', 'CBlock'], 'Other')
test_data['Exterior1st'] = test_data['Exterior1st'].replace(['WdShing', 'Stucco', 'AsbShng', 'BrkComm', 'Stone', 'AsphShn', 'ImStucc', 'CBlock'], 'Other')
train_data['Exterior1st'].value_counts()
exterior_mapping = {'VinylSd': 0, 'MetalSd': 1, 'HdBoard': 2, 'Wd Sdng': 3, 'Plywood': 4, 'CemntBd': 5, 'BrkFace': 6, 'Other': 7}
train_data['Exterior1st'] = train_data['Exterior1st'].map(exterior_mapping)
test_data['Exterior1st'] = test_data['Exterior1st'].map(exterior_mapping)
exterqual_mapping = {'TA': 0, 'Gd': 1, 'Ex': 2, 'Fa': 3}
train_data['ExterQual'] = train_data['ExterQual'].map(exterqual_mapping)
test_data['ExterQual'] = test_data['ExterQual'].map(exterqual_mapping)
paved_mapping = {'P': 0, 'N': 1, 'Y': 2}
train_data['PavedDrive'] = train_data['PavedDrive'].map(paved_mapping)
test_data['PavedDrive'] = test_data['PavedDrive'].map(paved_mapping)
found_mapping = {'PConc': 0, 'CBlock': 1, 'BrkTil': 2, 'Slab': 3, 'Stone': 4, 'Wood': 5}
train_data['Foundation'] = train_data['Foundation'].map(found_mapping)
test_data['Foundation'] = test_data['Foundation'].map(found_mapping)
bsmtqual_mapping = {'TA': 0, 'Gd': 1, 'Ex': 2, 'None': 3, 'Fa': 4}
train_data['BsmtQual'] = train_data['BsmtQual'].map(bsmtqual_mapping)
test_data['BsmtQual'] = test_data['BsmtQual'].map(bsmtqual_mapping)
kitchqual_mapping = {'TA': 0, 'Gd': 1, 'Ex': 2, 'Fa': 3}
train_data['KitchenQual'] = train_data['KitchenQual'].map(kitchqual_mapping)
test_data['KitchenQual'] = test_data['KitchenQual'].map(kitchqual_mapping)
train_data['Year_built_range'] = pd.cut(train_data['YearBuilt'], 10)
train_data['Year_built_range'].value_counts()
train_data.loc[train_data['YearBuilt'] <= 1899.6, 'YearBuilt'] = 0
train_data.loc[(train_data['YearBuilt'] > 1899.6) & (train_data['YearBuilt'] <= 1913.4), 'YearBuilt'] = 1
train_data.loc[(train_data['YearBuilt'] > 1913.4) & (train_data['YearBuilt'] <= 1927.2), 'YearBuilt'] = 2
train_data.loc[(train_data['YearBuilt'] > 1927.2) & (train_data['YearBuilt'] <= 1941), 'YearBuilt'] = 3
train_data.loc[(train_data['YearBuilt'] > 1941) & (train_data['YearBuilt'] <= 1954.8), 'YearBuilt'] = 4
train_data.loc[(train_data['YearBuilt'] > 1954.8) & (train_data['YearBuilt'] <= 1968.6), 'YearBuilt'] = 5
train_data.loc[(train_data['YearBuilt'] > 1968.6) & (train_data['YearBuilt'] <= 1982.4), 'YearBuilt'] = 6
train_data.loc[(train_data['YearBuilt'] > 1982.4) & (train_data['YearBuilt'] <= 1996.2), 'YearBuilt'] = 7
train_data.loc[train_data['YearBuilt'] > 1996.2, 'YearBuilt'] = 8
train_data['YearBuilt'].value_counts()
test_data.loc[test_data['YearBuilt'] <= 1899.6, 'YearBuilt'] = 0
test_data.loc[(test_data['YearBuilt'] > 1899.6) & (test_data['YearBuilt'] <= 1913.4), 'YearBuilt'] = 1
test_data.loc[(test_data['YearBuilt'] > 1913.4) & (test_data['YearBuilt'] <= 1927.2), 'YearBuilt'] = 2
test_data.loc[(test_data['YearBuilt'] > 1927.2) & (test_data['YearBuilt'] <= 1941), 'YearBuilt'] = 3
test_data.loc[(test_data['YearBuilt'] > 1941) & (test_data['YearBuilt'] <= 1954.8), 'YearBuilt'] = 4
test_data.loc[(test_data['YearBuilt'] > 1954.8) & (test_data['YearBuilt'] <= 1968.6), 'YearBuilt'] = 5
test_data.loc[(test_data['YearBuilt'] > 1968.6) & (test_data['YearBuilt'] <= 1982.4), 'YearBuilt'] = 6
test_data.loc[(test_data['YearBuilt'] > 1982.4) & (test_data['YearBuilt'] <= 1996.2), 'YearBuilt'] = 7
test_data.loc[test_data['YearBuilt'] > 1996.2, 'YearBuilt'] = 8
test_data['YearBuilt'].value_counts()
train_data['YearRemodAdd_range'] = pd.cut(train_data['YearRemodAdd'], 10)
train_data['YearRemodAdd_range'].value_counts()
test_data['YearRemodAdd_range'] = pd.cut(test_data['YearRemodAdd'], 10)
test_data['YearRemodAdd_range'].value_counts()
train_data.loc[train_data['YearRemodAdd'] <= 1956, 'YearRemodAdd'] = 0
train_data.loc[(train_data['YearRemodAdd'] > 1956) & (train_data['YearRemodAdd'] <= 1962), 'YearRemodAdd'] = 1
train_data.loc[(train_data['YearRemodAdd'] > 1962) & (train_data['YearRemodAdd'] <= 1968), 'YearRemodAdd'] = 2
train_data.loc[(train_data['YearRemodAdd'] > 1968) & (train_data['YearRemodAdd'] <= 1974), 'YearRemodAdd'] = 3
train_data.loc[(train_data['YearRemodAdd'] > 1974) & (train_data['YearRemodAdd'] <= 1980), 'YearRemodAdd'] = 4
train_data.loc[(train_data['YearRemodAdd'] > 1980) & (train_data['YearRemodAdd'] <= 1986), 'YearRemodAdd'] = 5
train_data.loc[(train_data['YearRemodAdd'] > 1986) & (train_data['YearRemodAdd'] <= 1992), 'YearRemodAdd'] = 6
train_data.loc[(train_data['YearRemodAdd'] > 1992) & (train_data['YearRemodAdd'] <= 1998), 'YearRemodAdd'] = 7
train_data.loc[train_data['YearRemodAdd'] > 1998, 'YearRemodAdd'] = 8
test_data.loc[test_data['YearRemodAdd'] <= 1956, 'YearRemodAdd'] = 0
test_data.loc[(test_data['YearRemodAdd'] > 1956) & (test_data['YearRemodAdd'] <= 1962), 'YearRemodAdd'] = 1
test_data.loc[(test_data['YearRemodAdd'] > 1962) & (test_data['YearRemodAdd'] <= 1968), 'YearRemodAdd'] = 2
test_data.loc[(test_data['YearRemodAdd'] > 1968) & (test_data['YearRemodAdd'] <= 1974), 'YearRemodAdd'] = 3
test_data.loc[(test_data['YearRemodAdd'] > 1974) & (test_data['YearRemodAdd'] <= 1980), 'YearRemodAdd'] = 4
test_data.loc[(test_data['YearRemodAdd'] > 1980) & (test_data['YearRemodAdd'] <= 1986), 'YearRemodAdd'] = 5
test_data.loc[(test_data['YearRemodAdd'] > 1986) & (test_data['YearRemodAdd'] <= 1992), 'YearRemodAdd'] = 6
test_data.loc[(test_data['YearRemodAdd'] > 1992) & (test_data['YearRemodAdd'] <= 1998), 'YearRemodAdd'] = 7
test_data.loc[test_data['YearRemodAdd'] > 1998, 'YearRemodAdd'] = 8
train_data = train_data.drop(['Year_built_range', 'YearRemodAdd_range'], axis=1)
test_data = test_data.drop('YearRemodAdd_range', axis=1)
yrsold_mapping = {2006: 0, 2007: 1, 2008: 2, 2009: 3, 2010: 4}
train_data['YrSold'] = train_data['YrSold'].map(yrsold_mapping)
test_data['YrSold'] = test_data['YrSold'].map(yrsold_mapping)
test_data['YrSold'].value_counts()
train_data['LotFrontage_range'] = pd.cut(train_data['LotFrontage'], 10)
train_data['LotFrontage_range'].value_counts()
train_data.loc[train_data['LotFrontage'] <= 50.2, 'LotFrontage'] = 0
train_data.loc[(train_data['LotFrontage'] > 50.2) & (train_data['LotFrontage'] <= 79.4), 'LotFrontage'] = 1
train_data.loc[(train_data['LotFrontage'] > 79.4) & (train_data['LotFrontage'] <= 108.6), 'LotFrontage'] = 2
train_data.loc[(train_data['LotFrontage'] > 108.6) & (train_data['LotFrontage'] <= 137.8), 'LotFrontage'] = 3
train_data.loc[(train_data['LotFrontage'] > 137.8) & (train_data['LotFrontage'] <= 167), 'LotFrontage'] = 4
train_data.loc[(train_data['LotFrontage'] > 167) & (train_data['LotFrontage'] <= 196.2), 'LotFrontage'] = 5
train_data.loc[(train_data['LotFrontage'] > 196.2) & (train_data['LotFrontage'] <= 225.4), 'LotFrontage'] = 6
train_data.loc[(train_data['LotFrontage'] > 225.4) & (train_data['LotFrontage'] <= 254.6), 'LotFrontage'] = 7
train_data.loc[train_data['LotFrontage'] > 254.6, 'LotFrontage'] = 8
test_data.loc[test_data['LotFrontage'] <= 50.2, 'LotFrontage'] = 0
test_data.loc[(test_data['LotFrontage'] > 50.2) & (test_data['LotFrontage'] <= 79.4), 'LotFrontage'] = 1
test_data.loc[(test_data['LotFrontage'] > 79.4) & (test_data['LotFrontage'] <= 108.6), 'LotFrontage'] = 2
test_data.loc[(test_data['LotFrontage'] > 108.6) & (test_data['LotFrontage'] <= 137.8), 'LotFrontage'] = 3
test_data.loc[(test_data['LotFrontage'] > 137.8) & (test_data['LotFrontage'] <= 167), 'LotFrontage'] = 4
test_data.loc[(test_data['LotFrontage'] > 167) & (test_data['LotFrontage'] <= 196.2), 'LotFrontage'] = 5
test_data.loc[(test_data['LotFrontage'] > 196.2) & (test_data['LotFrontage'] <= 225.4), 'LotFrontage'] = 6
test_data.loc[(test_data['LotFrontage'] > 225.4) & (test_data['LotFrontage'] <= 254.6), 'LotFrontage'] = 7
test_data.loc[test_data['LotFrontage'] > 254.6, 'LotFrontage'] = 8
print(train_data['LotFrontage'].value_counts())
print(test_data['LotFrontage'].value_counts())
train_data = train_data.drop('LotFrontage_range', axis=1)
train_data['TotalBsmtSF_range'] = pd.cut(train_data['TotalBsmtSF'], 10)
train_data['TotalBsmtSF_range'].value_counts()
train_data.loc[train_data['TotalBsmtSF'] <= 611, 'TotalBsmtSF'] = 0
train_data.loc[(train_data['TotalBsmtSF'] > 611) & (train_data['TotalBsmtSF'] <= 1222), 'TotalBsmtSF'] = 1
train_data.loc[(train_data['TotalBsmtSF'] > 1222) & (train_data['TotalBsmtSF'] <= 1833), 'TotalBsmtSF'] = 2
train_data.loc[(train_data['TotalBsmtSF'] > 1833) & (train_data['TotalBsmtSF'] <= 2444), 'TotalBsmtSF'] = 3
train_data.loc[(train_data['TotalBsmtSF'] > 2444) & (train_data['TotalBsmtSF'] <= 3055), 'TotalBsmtSF'] = 4
train_data.loc[(train_data['TotalBsmtSF'] > 3055) & (train_data['TotalBsmtSF'] <= 3666), 'TotalBsmtSF'] = 5
train_data.loc[(train_data['TotalBsmtSF'] > 3666) & (train_data['TotalBsmtSF'] <= 4277), 'TotalBsmtSF'] = 6
train_data.loc[(train_data['TotalBsmtSF'] > 4277) & (train_data['TotalBsmtSF'] <= 4888), 'TotalBsmtSF'] = 7
train_data.loc[train_data['TotalBsmtSF'] > 4888, 'TotalBsmtSF'] = 8
test_data.loc[test_data['TotalBsmtSF'] <= 611, 'TotalBsmtSF'] = 0
test_data.loc[(test_data['TotalBsmtSF'] > 611) & (test_data['TotalBsmtSF'] <= 1222), 'TotalBsmtSF'] = 1
test_data.loc[(test_data['TotalBsmtSF'] > 1222) & (test_data['TotalBsmtSF'] <= 1833), 'TotalBsmtSF'] = 2
test_data.loc[(test_data['TotalBsmtSF'] > 1833) & (test_data['TotalBsmtSF'] <= 2444), 'TotalBsmtSF'] = 3
test_data.loc[(test_data['TotalBsmtSF'] > 2444) & (test_data['TotalBsmtSF'] <= 3055), 'TotalBsmtSF'] = 4
test_data.loc[(test_data['TotalBsmtSF'] > 3055) & (test_data['TotalBsmtSF'] <= 3666), 'TotalBsmtSF'] = 5
test_data.loc[(test_data['TotalBsmtSF'] > 3666) & (test_data['TotalBsmtSF'] <= 4277), 'TotalBsmtSF'] = 6
test_data.loc[(test_data['TotalBsmtSF'] > 4277) & (test_data['TotalBsmtSF'] <= 4888), 'TotalBsmtSF'] = 7
test_data.loc[test_data['TotalBsmtSF'] > 4888, 'TotalBsmtSF'] = 8
train_data = train_data.drop('TotalBsmtSF_range', axis=1)
train_data['GrLivArea_range'] = pd.cut(train_data['GrLivArea'], 10)
train_data.loc[train_data['GrLivArea'] <= 864.8, 'GrLivArea'] = 0
train_data.loc[(train_data['GrLivArea'] > 864.8) & (train_data['GrLivArea'] <= 1395.6), 'GrLivArea'] = 1
train_data.loc[(train_data['GrLivArea'] > 1395.6) & (train_data['GrLivArea'] <= 1926.4), 'GrLivArea'] = 2
train_data.loc[(train_data['GrLivArea'] > 1926.4) & (train_data['GrLivArea'] <= 2457.2), 'GrLivArea'] = 3
train_data.loc[(train_data['GrLivArea'] > 2457.2) & (train_data['GrLivArea'] <= 2988), 'GrLivArea'] = 4
train_data.loc[(train_data['GrLivArea'] > 2988) & (train_data['GrLivArea'] <= 3518.8), 'GrLivArea'] = 5
train_data.loc[(train_data['GrLivArea'] > 3518.8) & (train_data['GrLivArea'] <= 4049.6), 'GrLivArea'] = 6
train_data.loc[(train_data['GrLivArea'] > 4049.6) & (train_data['GrLivArea'] <= 4580.4), 'GrLivArea'] = 7
train_data.loc[train_data['GrLivArea'] > 4580.4, 'GrLivArea'] = 8
test_data.loc[test_data['GrLivArea'] <= 864.8, 'GrLivArea'] = 0
test_data.loc[(test_data['GrLivArea'] > 864.8) & (test_data['GrLivArea'] <= 1395.6), 'GrLivArea'] = 1
test_data.loc[(test_data['GrLivArea'] > 1395.6) & (test_data['GrLivArea'] <= 1926.4), 'GrLivArea'] = 2
test_data.loc[(test_data['GrLivArea'] > 1926.4) & (test_data['GrLivArea'] <= 2457.2), 'GrLivArea'] = 3
test_data.loc[(test_data['GrLivArea'] > 2457.2) & (test_data['GrLivArea'] <= 2988), 'GrLivArea'] = 4
test_data.loc[(test_data['GrLivArea'] > 2988) & (test_data['GrLivArea'] <= 3518.8), 'GrLivArea'] = 5
test_data.loc[(test_data['GrLivArea'] > 3518.8) & (test_data['GrLivArea'] <= 4049.6), 'GrLivArea'] = 6
test_data.loc[(test_data['GrLivArea'] > 4049.6) & (test_data['GrLivArea'] <= 4580.4), 'GrLivArea'] = 7
test_data.loc[test_data['GrLivArea'] > 4580.4, 'GrLivArea'] = 8
print(train_data['GrLivArea'].value_counts())
print(test_data['GrLivArea'].value_counts())
train_data = train_data.drop('GrLivArea_range', axis=1)
train_data['WoodDeckSF_range'] = pd.cut(train_data['WoodDeckSF'], 10)
train_data['WoodDeckSF_range'].value_counts()
train_data.loc[train_data['WoodDeckSF'] <= 85.7, 'WoodDeckSF'] = 0
train_data.loc[(train_data['WoodDeckSF'] > 85.7) & (train_data['WoodDeckSF'] <= 171.4), 'WoodDeckSF'] = 1
train_data.loc[(train_data['WoodDeckSF'] > 171.4) & (train_data['WoodDeckSF'] <= 257.1), 'WoodDeckSF'] = 2
train_data.loc[(train_data['WoodDeckSF'] > 257.1) & (train_data['WoodDeckSF'] <= 342.8), 'WoodDeckSF'] = 3
train_data.loc[(train_data['WoodDeckSF'] > 342.8) & (train_data['WoodDeckSF'] <= 428.5), 'WoodDeckSF'] = 4
train_data.loc[(train_data['WoodDeckSF'] > 428.5) & (train_data['WoodDeckSF'] <= 514.2), 'WoodDeckSF'] = 5
train_data.loc[(train_data['WoodDeckSF'] > 514.2) & (train_data['WoodDeckSF'] <= 599.9), 'WoodDeckSF'] = 6
train_data.loc[(train_data['WoodDeckSF'] > 599.9) & (train_data['WoodDeckSF'] <= 685.6), 'WoodDeckSF'] = 7
train_data.loc[train_data['WoodDeckSF'] > 685.6, 'WoodDeckSF'] = 8
test_data.loc[test_data['WoodDeckSF'] <= 85.7, 'WoodDeckSF'] = 0
test_data.loc[(test_data['WoodDeckSF'] > 85.7) & (test_data['WoodDeckSF'] <= 171.4), 'WoodDeckSF'] = 1
test_data.loc[(test_data['WoodDeckSF'] > 171.4) & (test_data['WoodDeckSF'] <= 257.1), 'WoodDeckSF'] = 2
test_data.loc[(test_data['WoodDeckSF'] > 257.1) & (test_data['WoodDeckSF'] <= 342.8), 'WoodDeckSF'] = 3
test_data.loc[(test_data['WoodDeckSF'] > 342.8) & (test_data['WoodDeckSF'] <= 428.5), 'WoodDeckSF'] = 4
test_data.loc[(test_data['WoodDeckSF'] > 428.5) & (test_data['WoodDeckSF'] <= 514.2), 'WoodDeckSF'] = 5
test_data.loc[(test_data['WoodDeckSF'] > 514.2) & (test_data['WoodDeckSF'] <= 599.9), 'WoodDeckSF'] = 6
test_data.loc[(test_data['WoodDeckSF'] > 599.9) & (test_data['WoodDeckSF'] <= 685.6), 'WoodDeckSF'] = 7
test_data.loc[test_data['WoodDeckSF'] > 685.6, 'WoodDeckSF'] = 8
test_data['WoodDeckSF'].value_counts()
train_data = train_data.drop('WoodDeckSF_range', axis=1)
train_data['OpenPorchSF'].value_counts()
train_data = train_data.drop('3SsnPorch', axis=1)
test_data = test_data.drop('3SsnPorch', axis=1)
train_data = train_data.drop('ScreenPorch', axis=1)
test_data = test_data.drop('ScreenPorch', axis=1)
train_data['OpenPorchSF_range'] = pd.cut(train_data['OpenPorchSF'], 10)
train_data['OpenPorchSF_range'].value_counts()
train_data.loc[train_data['OpenPorchSF'] <= 54.7, 'OpenPorchSF'] = 0
train_data.loc[(train_data['OpenPorchSF'] > 54.7) & (train_data['OpenPorchSF'] <= 109.4), 'OpenPorchSF'] = 1
train_data.loc[(train_data['OpenPorchSF'] > 109.4) & (train_data['OpenPorchSF'] <= 164.1), 'OpenPorchSF'] = 2
train_data.loc[(train_data['OpenPorchSF'] > 164.1) & (train_data['OpenPorchSF'] <= 218.8), 'OpenPorchSF'] = 3
train_data.loc[(train_data['OpenPorchSF'] > 218.8) & (train_data['OpenPorchSF'] <= 273.5), 'OpenPorchSF'] = 4
train_data.loc[(train_data['OpenPorchSF'] > 273.5) & (train_data['OpenPorchSF'] <= 328.2), 'OpenPorchSF'] = 5
train_data.loc[(train_data['OpenPorchSF'] > 328.2) & (train_data['OpenPorchSF'] <= 382.9), 'OpenPorchSF'] = 6
train_data.loc[(train_data['OpenPorchSF'] > 382.9) & (train_data['OpenPorchSF'] <= 437.6), 'OpenPorchSF'] = 7
train_data.loc[train_data['OpenPorchSF'] > 437.6, 'OpenPorchSF'] = 8
test_data.loc[test_data['OpenPorchSF'] <= 54.7, 'OpenPorchSF'] = 0
test_data.loc[(test_data['OpenPorchSF'] > 54.7) & (test_data['OpenPorchSF'] <= 109.4), 'OpenPorchSF'] = 1
test_data.loc[(test_data['OpenPorchSF'] > 109.4) & (test_data['OpenPorchSF'] <= 164.1), 'OpenPorchSF'] = 2
test_data.loc[(test_data['OpenPorchSF'] > 164.1) & (test_data['OpenPorchSF'] <= 218.8), 'OpenPorchSF'] = 3
test_data.loc[(test_data['OpenPorchSF'] > 218.8) & (test_data['OpenPorchSF'] <= 273.5), 'OpenPorchSF'] = 4
test_data.loc[(test_data['OpenPorchSF'] > 273.5) & (test_data['OpenPorchSF'] <= 328.2), 'OpenPorchSF'] = 5
test_data.loc[(test_data['OpenPorchSF'] > 328.2) & (test_data['OpenPorchSF'] <= 382.9), 'OpenPorchSF'] = 6
test_data.loc[(test_data['OpenPorchSF'] > 382.9) & (test_data['OpenPorchSF'] <= 437.6), 'OpenPorchSF'] = 7
test_data.loc[test_data['OpenPorchSF'] > 437.6, 'OpenPorchSF'] = 8
train_data.head()
train_data = train_data.drop('OpenPorchSF_range', axis=1)
test_data.head()
for (counts, column) in enumerate(train_data.columns):
    print(counts, column + ':', train_data[column].isnull().sum(), train_data[column].dtype)
for (counts, column) in enumerate(test_data.columns):
    print(counts, column + ':', test_data[column].isnull().sum(), test_data[column].dtype)
train_data.head()
X_train = train_data.drop(['Id', 'SalePrice', 'LogSalePrice'], axis=1)
y_train = train_data['LogSalePrice']
X_test = test_data.drop('Id', axis=1)
lr = LinearRegression()