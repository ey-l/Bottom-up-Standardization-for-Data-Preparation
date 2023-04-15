import numpy as np
import pandas as pd
from scipy.stats import norm, skew
from scipy.special import boxcox1p
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import xticks

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df_train.head()
df_train.shape
df_train.describe().T
df_train.isnull().sum()
train_df = df_train.copy()
train_corr = df_train.corr()
plt.figure(figsize=(24, 24))
sns.heatmap(train_corr, annot=True, cmap='plasma')

top_50_corr = train_corr.index[abs(train_corr['SalePrice'] > 0.5)]
plt.figure(figsize=(12, 8))
top_50_corr = df_train[top_50_corr].corr()
sns.heatmap(top_50_corr, cmap='YlGnBu', annot=True)

corrmat = df_train.corr()
(f, ax) = plt.subplots(figsize=(15, 15))
sns.heatmap(corrmat, vmin=0, vmax=1, square=True, cmap='plasma', annot=True, mask=corrmat < 0.75, fmt='.1f', linecolor='black', center=0, annot_kws={'size': 7})

plt.figure(figsize=(18, 12))
plt.subplot(2, 2, 1)
sns.countplot(x='MSSubClass', data=df_train)
plt.subplot(2, 2, 2)
sns.countplot(x='MSZoning', data=df_train)
plt.subplot(2, 2, 3)
sns.countplot(x='Street', data=df_train)
plt.subplot(2, 2, 4)
sns.countplot(x='Alley', data=df_train)

plt.figure(figsize=(18, 12))
plt.subplot(2, 2, 1)
sns.countplot(x='LotShape', data=df_train)
plt.subplot(2, 2, 2)
sns.countplot(x='LandContour', data=df_train)
plt.subplot(2, 2, 3)
sns.countplot(x='Utilities', data=df_train)
plt.subplot(2, 2, 4)
sns.countplot(x='LotConfig', data=df_train)

plt.figure(figsize=(18, 12))
plt.subplot(2, 2, 1)
sns.countplot(x='LandSlope', data=df_train)
plt.subplot(2, 2, 2)
sns.countplot(x='Neighborhood', data=df_train)
plt.xticks(rotation=90)
plt.subplot(2, 2, 3)
sns.countplot(x='Condition1', data=df_train)
plt.subplot(2, 2, 4)
sns.countplot(x='Condition2', data=df_train)

plt.figure(figsize=(18, 12))
plt.subplot(2, 2, 1)
sns.countplot(x='RoofStyle', data=df_train)
plt.subplot(2, 2, 2)
sns.countplot(x='RoofMatl', data=df_train)
plt.subplot(2, 2, 3)
sns.countplot(x='Exterior1st', data=df_train)
plt.xticks(rotation=90)
plt.subplot(2, 2, 4)
sns.countplot(x='Exterior2nd', data=df_train)
plt.xticks(rotation=90)

plt.figure(figsize=(18, 12))
plt.subplot(2, 2, 1)
sns.countplot(x='MasVnrType', data=df_train)
plt.subplot(2, 2, 2)
sns.countplot(x='ExterQual', data=df_train)
plt.subplot(2, 2, 3)
sns.countplot(x='ExterCond', data=df_train)
plt.subplot(2, 2, 4)
sns.countplot(x='Foundation', data=df_train)

plt.figure(figsize=(18, 12))
plt.subplot(2, 2, 1)
sns.countplot(x='BsmtQual', data=df_train)
plt.subplot(2, 2, 2)
sns.countplot(x='BsmtCond', data=df_train)
plt.subplot(2, 2, 3)
sns.countplot(x='BsmtExposure', data=df_train)
plt.subplot(2, 2, 4)
sns.countplot(x='BsmtFinType1', data=df_train)

plt.figure(figsize=(18, 12))
plt.subplot(2, 2, 1)
sns.countplot(x='GarageType', data=df_train)
plt.subplot(2, 2, 2)
sns.countplot(x='GarageFinish', data=df_train)
plt.subplot(2, 2, 3)
sns.countplot(x='GarageQual', data=df_train)
plt.subplot(2, 2, 4)
sns.countplot(x='GarageCond', data=df_train)

plt.figure(figsize=(18, 12))
plt.subplot(1, 2, 1)
sns.countplot(x='SaleType', data=df_train)
plt.subplot(1, 2, 2)
sns.countplot(x='SaleCondition', data=df_train)

numeric_variable = df_train.select_dtypes(['float64', 'int64'])
plt.figure(figsize=(15, 10))
numeric_variable.groupby(['YearBuilt'])['SalePrice'].mean().plot(kind='line')
plt.ylabel('Mean Sales Price')

plt.figure(figsize=(14, 8))
plt.subplot(2, 2, 1)
sns.distplot(df_train['BsmtFinSF1'])
plt.subplot(2, 2, 2)
sns.distplot(df_train['BsmtFinSF2'])
plt.subplot(2, 2, 3)
sns.distplot(df_train['BsmtUnfSF'])
plt.subplot(2, 2, 4)
sns.distplot(df_train['TotalBsmtSF'])

df_train.drop(['BsmtFinSF2'], axis=1, inplace=True)
plt.figure(figsize=(19, 5))
plt.subplot(1, 3, 1)
sns.scatterplot(x='BsmtFinSF1', y='SalePrice', data=numeric_variable)
plt.subplot(1, 3, 2)
sns.scatterplot(x='BsmtUnfSF', y='SalePrice', data=numeric_variable)
plt.subplot(1, 3, 3)
sns.scatterplot(x='TotalBsmtSF', y='SalePrice', data=numeric_variable)

plt.figure(figsize=(14, 8))
plt.subplot(2, 2, 1)
sns.distplot(numeric_variable['1stFlrSF'])
plt.subplot(2, 2, 2)
sns.distplot(numeric_variable['2ndFlrSF'])
plt.subplot(2, 2, 3)
sns.distplot(numeric_variable['LowQualFinSF'])
plt.subplot(2, 2, 4)
sns.distplot(numeric_variable['GrLivArea'])

df_train.drop(['LowQualFinSF'], axis=1, inplace=True)
plt.figure(figsize=(16, 8))
plt.subplot(2, 2, 1)
sns.scatterplot(x='BsmtFullBath', y='SalePrice', data=numeric_variable)
plt.subplot(2, 2, 2)
sns.scatterplot(x='BsmtHalfBath', y='SalePrice', data=numeric_variable)
plt.subplot(2, 2, 3)
sns.scatterplot(x='FullBath', y='SalePrice', data=numeric_variable)
plt.subplot(2, 2, 4)
sns.scatterplot(x='HalfBath', y='SalePrice', data=numeric_variable)

plt.figure(figsize=(16, 8))
plt.subplot(2, 3, 1)
sns.distplot(numeric_variable['WoodDeckSF'])
plt.subplot(2, 3, 2)
sns.distplot(numeric_variable['OpenPorchSF'])
plt.subplot(2, 3, 3)
sns.distplot(numeric_variable['EnclosedPorch'])
plt.subplot(2, 3, 4)
sns.distplot(numeric_variable['3SsnPorch'])
plt.subplot(2, 3, 5)
sns.distplot(numeric_variable['ScreenPorch'])
plt.subplot(2, 3, 6)
sns.distplot(numeric_variable['PoolArea'])

df_train.drop(['EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea'], axis=1, inplace=True)
sns.scatterplot(x='MiscVal', y='SalePrice', data=numeric_variable)

df_train.drop(['MiscVal', 'Id', 'BedroomAbvGr', 'KitchenAbvGr'], axis=1, inplace=True)
count = pd.DataFrame(df_train.isnull().sum().sort_values(ascending=False), columns=['null_counts'])
percent = pd.DataFrame(round(100 * (df_train.isnull().sum() / df_train.shape[0]), 2).sort_values(ascending=False), columns=['null_percentage'])
Missing_Value_Table = pd.concat([count, percent], axis=1)
Missing_Value_Table.head(20)
high_percnt_missing_features = ['Alley', 'PoolQC', 'MiscFeature', 'Fence']
for miss in high_percnt_missing_features:
    print('Unique Values Count for {}'.format(miss))
    print(df_train[miss].value_counts(dropna=False))
    print('\n')
df_train.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)
df_train.shape
df_train['FireplaceQu'].value_counts(dropna=False)
df_train.loc[df_train['FireplaceQu'].isnull(), ['Fireplaces', 'FireplaceQu']].head(10)
df_train['FireplaceQu'].fillna('Not_Present', inplace=True)
sns.distplot(df_train['LotFrontage'].dropna())

df_train['LotFrontage'].fillna(df_train['LotFrontage'].mean(), inplace=True)
df_train.loc[df_train['GarageType'].isnull(), ['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond']].head(10)
df_train.loc[df_train.GarageType.isnull(), ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']] = 'Not_Present'
sns.distplot(df_train['GarageYrBlt'].dropna())

df_train['GarageYrBlt'].fillna(2021, inplace=True)
df_train.loc[df_train.BsmtQual.isnull(), ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']] = 'Not_Present'
df_train.isnull().sum()
df_train['BsmtExposure'].replace(np.nan, 'Not_Present', inplace=True)
df_train['BsmtFinType2'].replace(np.nan, 'Not_Present', inplace=True)
df_train.loc[df_train.MasVnrArea.isnull(), ['MasVnrArea', 'MasVnrType']]
df_train.loc[df_train.MasVnrType == 'None', ['MasVnrArea', 'MasVnrType']].head()
df_train['MasVnrType'].fillna('Not_Present', inplace=True)
df_train['MasVnrArea'].fillna(0, inplace=True)
df_train['Electrical'].value_counts(dropna=False)
df_train['Electrical'].fillna(df_train['Electrical'].mode()[0], inplace=True)
df_train.isnull().sum()
df_train['MSSubClass'] = df_train['MSSubClass'].apply(str)
df_train['OverallCond'] = df_train['OverallCond'].astype(str)
df_train['YrSold'] = df_train['YrSold'].astype(str)
df_train['MoSold'] = df_train['MoSold'].astype(str)
df_train['Street'].astype('category').value_counts()
df_train['Utilities'].astype('category').value_counts()
df_train.drop(['Street', 'Utilities'], axis=1, inplace=True)
plt.figure(figsize=(10, 6))
sns.distplot(df_train['SalePrice'])

plt.figure(figsize=(10, 6))
sns.distplot(np.log1p(df_train.SalePrice))

df_train['SalePrice'] = np.log1p(df_train['SalePrice'])
cat_cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 'ExterQual', 'ExterCond', 'HeatingQC', 'KitchenQual', 'BsmtFinType1', 'BsmtFinType2', 'Functional', 'BsmtExposure', 'GarageFinish', 'LandSlope', 'LotShape', 'PavedDrive', 'CentralAir', 'MSSubClass', 'OverallCond', 'YrSold', 'MoSold')
for col in cat_cols:
    lb = LabelEncoder()