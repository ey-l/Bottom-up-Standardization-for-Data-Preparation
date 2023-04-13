import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print(_input1.shape)
print(_input0.shape)
_input1.describe()
_input1.info()
plt.figure(figsize=(20, 8))
sns.heatmap(_input1.isnull(), cbar=False)
plt.figure(figsize=(20, 8))
sns.heatmap(_input1.isnull(), cbar=False)
pd.options.display.min_rows = 80
_input1.isnull().sum().sort_values(ascending=False)
_input1 = _input1.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage'], axis=1, inplace=False)
_input0 = _input0.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage'], axis=1, inplace=False)
_input1['GarageYrBlt'] = _input1['GarageYrBlt'].fillna(_input1['GarageYrBlt'].dropna().mode()[0], inplace=False)
_input1['GarageCond'] = _input1['GarageCond'].fillna(_input1['GarageCond'].dropna().mode()[0], inplace=False)
_input1['GarageType'] = _input1['GarageType'].fillna(_input1['GarageType'].dropna().mode()[0], inplace=False)
_input1['GarageFinish'] = _input1['GarageFinish'].fillna(_input1['GarageFinish'].dropna().mode()[0], inplace=False)
_input1['GarageQual'] = _input1['GarageQual'].fillna(_input1['GarageQual'].dropna().mode()[0], inplace=False)
_input1['BsmtFinType2'] = _input1['BsmtFinType2'].fillna(_input1['BsmtFinType2'].dropna().mode()[0], inplace=False)
_input1['BsmtExposure'] = _input1['BsmtExposure'].fillna(_input1['BsmtExposure'].dropna().mode()[0], inplace=False)
_input1['BsmtQual'] = _input1['BsmtQual'].fillna(_input1['BsmtQual'].dropna().mode()[0], inplace=False)
_input1['BsmtCond'] = _input1['BsmtCond'].fillna(_input1['BsmtCond'].dropna().mode()[0], inplace=False)
_input1['BsmtFinType1'] = _input1['BsmtFinType1'].fillna(_input1['BsmtFinType1'].dropna().mode()[0], inplace=False)
_input1['MasVnrArea'] = _input1['MasVnrArea'].fillna(_input1['MasVnrArea'].dropna().mode()[0], inplace=False)
_input1['MasVnrType'] = _input1['MasVnrType'].fillna(_input1['MasVnrType'].dropna().mode()[0], inplace=False)
_input1['Electrical'] = _input1['Electrical'].fillna(_input1['Electrical'].dropna().mode()[0], inplace=False)
pd.options.display.min_rows = 82
_input1.isnull().sum().sort_values(ascending=False)
_input0['GarageYrBlt'] = _input0['GarageYrBlt'].fillna(_input0['GarageYrBlt'].dropna().mode()[0], inplace=False)
_input0['GarageFinish'] = _input0['GarageFinish'].fillna(_input0['GarageFinish'].dropna().mode()[0], inplace=False)
_input0['GarageQual'] = _input0['GarageQual'].fillna(_input0['GarageQual'].dropna().mode()[0], inplace=False)
_input0['GarageCond'] = _input0['GarageCond'].fillna(_input0['GarageCond'].dropna().mode()[0], inplace=False)
_input0['GarageType'] = _input0['GarageType'].fillna(_input0['GarageType'].dropna().mode()[0], inplace=False)
_input0['BsmtCond'] = _input0['BsmtCond'].fillna(_input0['BsmtCond'].dropna().mode()[0], inplace=False)
_input0['BsmtExposure'] = _input0['BsmtExposure'].fillna(_input0['BsmtExposure'].dropna().mode()[0], inplace=False)
_input0['BsmtQual'] = _input0['BsmtQual'].fillna(_input0['BsmtQual'].dropna().mode()[0], inplace=False)
_input0['BsmtFinType1'] = _input0['BsmtFinType1'].fillna(_input0['BsmtFinType1'].dropna().mode()[0], inplace=False)
_input0['BsmtFinType2'] = _input0['BsmtFinType2'].fillna(_input0['BsmtFinType2'].dropna().mode()[0], inplace=False)
_input0['MasVnrType'] = _input0['MasVnrType'].fillna(_input0['MasVnrType'].dropna().mode()[0], inplace=False)
_input0['MasVnrArea'] = _input0['MasVnrArea'].fillna(_input0['MasVnrArea'].dropna().mode()[0], inplace=False)
_input0['MSZoning'] = _input0['MSZoning'].fillna(_input0['MSZoning'].dropna().mode()[0], inplace=False)
_input0['BsmtFullBath'] = _input0['BsmtFullBath'].fillna(_input0['BsmtFullBath'].dropna().mode()[0], inplace=False)
_input0['Utilities'] = _input0['Utilities'].fillna(_input0['Utilities'].dropna().mode()[0], inplace=False)
_input0['Functional'] = _input0['Functional'].fillna(_input0['Functional'].dropna().mode()[0], inplace=False)
_input0['BsmtHalfBath'] = _input0['BsmtHalfBath'].fillna(_input0['BsmtHalfBath'].dropna().mode()[0], inplace=False)
_input0['BsmtFinSF1'] = _input0['BsmtFinSF1'].fillna(_input0['BsmtFinSF1'].dropna().mean(), inplace=False)
_input0['BsmtFinSF2'] = _input0['BsmtFinSF2'].fillna(_input0['BsmtFinSF2'].dropna().mean(), inplace=False)
_input0['BsmtUnfSF'] = _input0['BsmtUnfSF'].fillna(_input0['BsmtUnfSF'].dropna().mean(), inplace=False)
_input0['TotalBsmtSF'] = _input0['TotalBsmtSF'].fillna(_input0['TotalBsmtSF'].dropna().mean(), inplace=False)
_input0['KitchenQual'] = _input0['KitchenQual'].fillna(_input0['KitchenQual'].dropna().mode()[0], inplace=False)
_input0['Exterior2nd'] = _input0['Exterior2nd'].fillna(_input0['Exterior2nd'].dropna().mode()[0], inplace=False)
_input0['Exterior1st'] = _input0['Exterior1st'].fillna(_input0['Exterior1st'].dropna().mode()[0], inplace=False)
_input0['GarageArea'] = _input0['GarageArea'].fillna(_input0['GarageArea'].dropna().mean(), inplace=False)
_input0['SaleType'] = _input0['SaleType'].fillna(_input0['SaleType'].dropna().mode()[0], inplace=False)
_input0['GarageCars'] = _input0['GarageCars'].fillna(_input0['GarageCars'].dropna().mean(), inplace=False)
_input0.isnull().sum().sort_values(ascending=False)
_input0.info()
_input0.info()
print(_input1.shape)
print(_input0.shape)
_input1.corr()
plt.figure(figsize=(30, 18))
sns.heatmap(_input1.corr(), center=0, annot=True)
df = pd.concat([_input1, _input0], axis=0)
df

def category(object_columns):
    df_final = df
    i = 0
    for fields in object_columns:
        print(fields)
        df1 = pd.get_dummies(df[fields], drop_first=True)
        df = df.drop([fields], axis=1, inplace=False)
        if i == 0:
            df_final = df1.copy()
        else:
            df_final = pd.concat([df_final, df1], axis=1)
        i += 1
    df_final = pd.concat([df, df_final], axis=1)
    return df_final
categorical_feature = [feature for feature in _input1.columns if _input1[feature].dtype == 'O']
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
test_df = test_df.drop(['SalePrice'], axis=1, inplace=False)
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