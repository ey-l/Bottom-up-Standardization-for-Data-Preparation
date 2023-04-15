import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print(df_train.shape)
print(df_test.shape)
df_train.describe()
df_train.info()
plt.figure(figsize=(20, 8))
sns.heatmap(df_train.isnull(), cbar=False)
plt.figure(figsize=(20, 8))
sns.heatmap(df_train.isnull(), cbar=False)
pd.options.display.min_rows = 80
df_train.isnull().sum().sort_values(ascending=False)
df_train.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage'], axis=1, inplace=True)
df_test.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage'], axis=1, inplace=True)
df_train['GarageYrBlt'].fillna(df_train['GarageYrBlt'].dropna().mode()[0], inplace=True)
df_train['GarageCond'].fillna(df_train['GarageCond'].dropna().mode()[0], inplace=True)
df_train['GarageType'].fillna(df_train['GarageType'].dropna().mode()[0], inplace=True)
df_train['GarageFinish'].fillna(df_train['GarageFinish'].dropna().mode()[0], inplace=True)
df_train['GarageQual'].fillna(df_train['GarageQual'].dropna().mode()[0], inplace=True)
df_train['BsmtFinType2'].fillna(df_train['BsmtFinType2'].dropna().mode()[0], inplace=True)
df_train['BsmtExposure'].fillna(df_train['BsmtExposure'].dropna().mode()[0], inplace=True)
df_train['BsmtQual'].fillna(df_train['BsmtQual'].dropna().mode()[0], inplace=True)
df_train['BsmtCond'].fillna(df_train['BsmtCond'].dropna().mode()[0], inplace=True)
df_train['BsmtFinType1'].fillna(df_train['BsmtFinType1'].dropna().mode()[0], inplace=True)
df_train['MasVnrArea'].fillna(df_train['MasVnrArea'].dropna().mode()[0], inplace=True)
df_train['MasVnrType'].fillna(df_train['MasVnrType'].dropna().mode()[0], inplace=True)
df_train['Electrical'].fillna(df_train['Electrical'].dropna().mode()[0], inplace=True)
pd.options.display.min_rows = 82
df_train.isnull().sum().sort_values(ascending=False)
df_test['GarageYrBlt'].fillna(df_test['GarageYrBlt'].dropna().mode()[0], inplace=True)
df_test['GarageFinish'].fillna(df_test['GarageFinish'].dropna().mode()[0], inplace=True)
df_test['GarageQual'].fillna(df_test['GarageQual'].dropna().mode()[0], inplace=True)
df_test['GarageCond'].fillna(df_test['GarageCond'].dropna().mode()[0], inplace=True)
df_test['GarageType'].fillna(df_test['GarageType'].dropna().mode()[0], inplace=True)
df_test['BsmtCond'].fillna(df_test['BsmtCond'].dropna().mode()[0], inplace=True)
df_test['BsmtExposure'].fillna(df_test['BsmtExposure'].dropna().mode()[0], inplace=True)
df_test['BsmtQual'].fillna(df_test['BsmtQual'].dropna().mode()[0], inplace=True)
df_test['BsmtFinType1'].fillna(df_test['BsmtFinType1'].dropna().mode()[0], inplace=True)
df_test['BsmtFinType2'].fillna(df_test['BsmtFinType2'].dropna().mode()[0], inplace=True)
df_test['MasVnrType'].fillna(df_test['MasVnrType'].dropna().mode()[0], inplace=True)
df_test['MasVnrArea'].fillna(df_test['MasVnrArea'].dropna().mode()[0], inplace=True)
df_test['MSZoning'].fillna(df_test['MSZoning'].dropna().mode()[0], inplace=True)
df_test['BsmtFullBath'].fillna(df_test['BsmtFullBath'].dropna().mode()[0], inplace=True)
df_test['Utilities'].fillna(df_test['Utilities'].dropna().mode()[0], inplace=True)
df_test['Functional'].fillna(df_test['Functional'].dropna().mode()[0], inplace=True)
df_test['BsmtHalfBath'].fillna(df_test['BsmtHalfBath'].dropna().mode()[0], inplace=True)
df_test['BsmtFinSF1'].fillna(df_test['BsmtFinSF1'].dropna().mean(), inplace=True)
df_test['BsmtFinSF2'].fillna(df_test['BsmtFinSF2'].dropna().mean(), inplace=True)
df_test['BsmtUnfSF'].fillna(df_test['BsmtUnfSF'].dropna().mean(), inplace=True)
df_test['TotalBsmtSF'].fillna(df_test['TotalBsmtSF'].dropna().mean(), inplace=True)
df_test['KitchenQual'].fillna(df_test['KitchenQual'].dropna().mode()[0], inplace=True)
df_test['Exterior2nd'].fillna(df_test['Exterior2nd'].dropna().mode()[0], inplace=True)
df_test['Exterior1st'].fillna(df_test['Exterior1st'].dropna().mode()[0], inplace=True)
df_test['GarageArea'].fillna(df_test['GarageArea'].dropna().mean(), inplace=True)
df_test['SaleType'].fillna(df_test['SaleType'].dropna().mode()[0], inplace=True)
df_test['GarageCars'].fillna(df_test['GarageCars'].dropna().mean(), inplace=True)
df_test.isnull().sum().sort_values(ascending=False)
df_test.info()
df_test.info()
print(df_train.shape)
print(df_test.shape)
df_train.corr()
plt.figure(figsize=(30, 18))
sns.heatmap(df_train.corr(), center=0, annot=True)
df = pd.concat([df_train, df_test], axis=0)
df

def category(object_columns):
    df_final = df
    i = 0
    for fields in object_columns:
        print(fields)
        df1 = pd.get_dummies(df[fields], drop_first=True)
        df.drop([fields], axis=1, inplace=True)
        if i == 0:
            df_final = df1.copy()
        else:
            df_final = pd.concat([df_final, df1], axis=1)
        i += 1
    df_final = pd.concat([df, df_final], axis=1)
    return df_final
categorical_feature = [feature for feature in df_train.columns if df_train[feature].dtype == 'O']
len(categorical_feature)
df = category(categorical_feature)
df.shape
train_df = df.iloc[:1460, :]
test_df = df.iloc[1460:, :]
print(train_df.shape)
print(test_df.shape)
sns.histplot(x='SalePrice', data=train_df, bins=20)
plt.hist(x='SalePrice', data=train_df, bins=20, rwidth=1.5, ec='black')
max_sellingprice = train_df['SalePrice'].max()
print('The Largest Sale Price :', max_sellingprice, 'usd')
max_YrSold = train_df['YrSold'].max()
print('Largest sale year  :', max_YrSold, 'usd')
columns = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train_df[columns])
train_df
test_df
test_df.drop(['SalePrice'], axis=1, inplace=True)
X_train = train_df.drop(['SalePrice'], axis=1)
Y_train = train_df['SalePrice']
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
test_df = sc.transform(test_df)
X_train
test_df
from sklearn.linear_model import LinearRegression
reg = LinearRegression()