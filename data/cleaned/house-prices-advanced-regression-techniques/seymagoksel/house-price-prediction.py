import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train.head()
print('TRAIN ROW SIZE:', train.shape[0], 'TRAIN COLUMN SIZE:', train.shape[1])
print('TEST ROW SIZE:', test.shape[0], 'TEST COLUMN SIZE:', test.shape[1])
test.head()
test.head()
saleprice = train['SalePrice']
saleprice
train.describe().T
train['SalePrice'].describe().T
train.info()
plt.figure(figsize=(9, 6))
sns.distplot(train['SalePrice'], color='r')
plt.ylabel('Frequency')
plt.title('SalePrice')
print('SKEWNESS: %f' % train['SalePrice'].skew())
print('KURTOSIS: %f' % train['SalePrice'].kurt())
train.corr()
plt.subplots(figsize=(15, 12))
sns.heatmap(train.corr(), vmax=1, cmap=sns.color_palette('Reds'), square=True)
corr_saleprice = pd.DataFrame(train.corrwith(saleprice, axis=0))
corr_saleprice.rename(columns={0: 'SalePrice'}, inplace=True)
plt.subplots(figsize=(15, 12))
sns.set(font_scale=1.1)
sns.heatmap(corr_saleprice, vmax=1, cmap=sns.color_palette('Reds'), fmt='.4f', annot=True)
data = pd.concat([train['SalePrice'], train['OverallQual']], axis=1)
plt.subplots(figsize=(8, 6))
sns.boxplot(x='OverallQual', y='SalePrice', data=data, palette='Reds').axis(ymin=0, ymax=800000)
plt.figure(figsize=(40, 20))
sns.set(font_scale=1.2)
sns.boxplot(x='YearBuilt', y='SalePrice', data=train, palette='Reds')
plt.xticks(rotation=90)
plt.subplots(figsize=(9, 7))
plt.scatter(train['TotalBsmtSF'], train['SalePrice'], color='#e88e76', marker='*')
plt.subplots(figsize=(9, 7))
plt.scatter(x=train['1stFlrSF'], y=train['SalePrice'], color='#e57760', marker='*', alpha=0.7)
plt.subplots(figsize=(9, 7))
plt.scatter(x=train['GrLivArea'], y=train['SalePrice'], color='#d85346', marker='*', alpha=0.7)
plt.subplots(figsize=(9, 7))
plt.scatter(x=train['FullBath'], y=train['SalePrice'], color='#c43e3b', marker='*', alpha=0.7)
plt.subplots(figsize=(9, 7))
plt.scatter(x=train['TotRmsAbvGrd'], y=train['SalePrice'], color='#b02c30', marker='*', alpha=0.7)
plt.subplots(figsize=(9, 7))
plt.scatter(x=train['GarageCars'], y=train['SalePrice'], color='#9a2529', marker='*', alpha=0.7)
plt.subplots(figsize=(9, 7))
plt.scatter(x=train['GarageArea'], y=train['SalePrice'], color='#7c1920', marker='*', alpha=0.7)
s = train.dtypes == 'object'
object_cols = list(s[s].index)
print('Categorical variables:')
print(object_cols)
for c in object_cols:
    plt.figure(figsize=(7, 4))
    plt.xticks(rotation=90)
    sns.countplot(x=train[c].to_numpy(), palette='Reds')

    print(train[c].value_counts())
    print('-' * 100)
train = train.drop('SalePrice', axis='columns')
train.shape
data = pd.concat([train, test])
data = data.reset_index(drop=True)
data.head()
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum() / train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Sum', 'Percent'])
missing_data = missing_data
missing_data.head(34)
missing_data
(f, ax) = plt.subplots(figsize=(15, 8))
missing = round(data.isnull().mean(), 5)
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar(color='#e57760')
ax.set(ylabel='PERCENT')
ax.set(xlabel='FIELDS')
sns.despine(trim=True, left=True)
data['Functional'] = data['Functional'].fillna('Typ')
data['Electrical'] = data['Electrical'].fillna('SBrkr')
data['KitchenQual'] = data['KitchenQual'].fillna('TA')
data['Exterior1st'] = data['Exterior1st'].fillna(data['Exterior1st'].mode()[0])
data['Exterior2nd'] = data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0])
data['SaleType'] = data['SaleType'].fillna(data['SaleType'].mode()[0])
data['MSZoning'] = data.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
data['PoolQC'] = data['PoolQC'].fillna('NA')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    data[col] = data[col].fillna(0)
for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    data[col] = data[col].fillna('NA')
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    data[col] = data[col].fillna('NA')
data['LotFrontage'] = data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
data.update(data[object_cols].fillna('NA'))
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric = []
for i in data.columns:
    if data[i].dtype in numeric_dtypes:
        numeric.append(i)
data.update(data[numeric].fillna(0))
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum() / train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Sum', 'Percent'])
missing_data[:20]
BsmtQual_ = data['BsmtQual']
BsmtQual = []
HasBsmt = []
for index in range(BsmtQual_.size):
    if BsmtQual_[index] == 'NA':
        BsmtQual.append(0)
        HasBsmt.append(0)
    else:
        HasBsmt.append(1)
    if BsmtQual_[index] == 'Ex':
        BsmtQual.append(105)
    if BsmtQual_[index] == 'Gd':
        BsmtQual.append(95)
    if BsmtQual_[index] == 'TA':
        BsmtQual.append(85)
    if BsmtQual_[index] == 'Fa':
        BsmtQual.append(75)
    if BsmtQual_[index] == 'Po':
        BsmtQual.append(65)
data['HasBsmt'] = HasBsmt
data['BsmtQual'] = BsmtQual
label_encoder = LabelEncoder()
for col in object_cols:
    data[col] = label_encoder.fit_transform(data[col])
data.head()
data['HasWoodDeck'] = (data['WoodDeckSF'] == 0) * 1
data['HasOpenPorch'] = (data['OpenPorchSF'] == 0) * 1
data['HasEnclosedPorch'] = (data['EnclosedPorch'] == 0) * 1
data['Has3SsnPorch'] = (data['3SsnPorch'] == 0) * 1
data['HasScreenPorch'] = (data['ScreenPorch'] == 0) * 1
data['YearsSinceRemodel'] = data['YrSold'].astype(int) - data['YearRemodAdd'].astype(int)
data['TotalHomeQuality'] = data['OverallQual'] + data['OverallCond']
data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
data['YrBltAndRemod'] = data['YearBuilt'] + data['YearRemodAdd']
data['TotalSQR'] = data['BsmtFinSF1'] + data['BsmtFinSF2'] + data['1stFlrSF'] + data['2ndFlrSF']
data['TotalBathrooms'] = data['FullBath'] + 0.5 * data['HalfBath'] + data['BsmtFullBath'] + 0.5 * data['BsmtHalfBath']
data['TotalPorchSf'] = data['OpenPorchSF'] + data['3SsnPorch'] + data['EnclosedPorch'] + data['ScreenPorch'] + data['WoodDeckSF']
data['HasPool'] = data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
data['Has2ndFloor'] = data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
data['HasGarage'] = data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
data['HasBsmt'] = data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
data['HasFireplace'] = data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
data.shape
cols = ['Street', 'Utilities', 'LandSlope', 'Condition1', 'Condition2', 'RoofStyle', 'RoofMatl', 'BsmtCond', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'MiscFeature', 'SaleType', 'SaleCondition']
data = data.drop(cols, axis=1)
data.shape
s = data.dtypes == 'object'
object_cols = list(s[s].index)
print('Categorical variables:')
print(object_cols)
train = data[:1460].drop('Id', axis='columns')
test = data[1460:].drop('Id', axis='columns')
print('TRAIN ROW SIZE:', train.shape[0], 'TRAIN COLUMN SIZE:', train.shape[1])
print('TEST ROW SIZE:', test.shape[0], 'TEST COLUMN SIZE:', test.shape[1])
scaler = StandardScaler()