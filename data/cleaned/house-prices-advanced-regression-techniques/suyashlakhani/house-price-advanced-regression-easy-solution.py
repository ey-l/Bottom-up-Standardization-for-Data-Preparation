import numpy as np
import pandas as pd
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
df.head()
sns.heatmap(df.isnull(), yticklabels=False, cbar=False)
df.isnull().sum()
df.info()
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean())
df.drop(['Alley'], axis=1, inplace=True)
df['MasVnrType'] = df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
df['MasVnrArea'] = df['MasVnrArea'].fillna(df['MasVnrArea'].mode()[0])
df['BsmtCond'] = df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
df['BsmtQual'] = df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
df['FireplaceQu'] = df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])
df['GarageType'] = df['GarageType'].fillna(df['GarageType'].mode()[0])
df['GarageFinish'] = df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])
df['GarageQual'] = df['GarageQual'].fillna(df['GarageQual'].mode()[0])
df['GarageCond'] = df['GarageCond'].fillna(df['GarageCond'].mode()[0])
df.drop(['GarageYrBlt'], axis=1, inplace=True)
df.drop(['PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)
df.shape
df.drop(['Id'], axis=1, inplace=True)
df.isnull().sum()
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='coolwarm')
df['BsmtExposure'] = df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])
df['BsmtFinType2'] = df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='YlGnBu')
df.dropna(inplace=True)
df.shape
cat_feature = [feature for feature in df.columns if df[feature].dtype == 'O']
len(cat_feature)

def category_onehot_multcols(multcolumns):
    df_final = final_df
    i = 0
    for fields in multcolumns:
        print(fields)
        df1 = pd.get_dummies(final_df[fields], drop_first=True)
        final_df.drop([fields], axis=1, inplace=True)
        if i == 0:
            df_final = df1.copy()
        else:
            df_final = pd.concat([df_final, df1], axis=1)
        i = i + 1
    df_final = pd.concat([final_df, df_final], axis=1)
    return df_final
main_df = df.copy()
test_df.head()
sol = test_df['Id']
test_df.shape
test_num = [feature for feature in test_df.columns if test_df[feature].isnull().sum() > 1]
test_num
test_df['MSZoning'] = test_df['MSZoning'].fillna(test_df['MSZoning'].mode()[0])
test_df['LotFrontage'] = test_df['LotFrontage'].fillna(test_df['LotFrontage'].mean())
test_df.drop(['Alley'], axis=1, inplace=True)
test_df.shape
test_df['Utilities'] = test_df['Utilities'].fillna(test_df['Utilities'].mode()[0])
test_df['MasVnrType'] = test_df['MasVnrType'].fillna(test_df['MasVnrType'].mode()[0])
test_df['MasVnrArea'] = test_df['MasVnrArea'].fillna(test_df['MasVnrArea'].mode()[0])
test_df['BsmtCond'] = test_df['BsmtCond'].fillna(test_df['BsmtCond'].mode()[0])
test_df['BsmtQual'] = test_df['BsmtQual'].fillna(test_df['BsmtQual'].mode()[0])
test_df['BsmtExposure'] = test_df['BsmtExposure'].fillna(test_df['BsmtExposure'].mode()[0])
test_df['BsmtFinType1'] = test_df['BsmtFinType1'].fillna(test_df['BsmtFinType1'].mode()[0])
test_df['BsmtFinType2'] = test_df['BsmtFinType2'].fillna(test_df['BsmtFinType2'].mode()[0])
test_df['BsmtFullBath'] = test_df['BsmtFullBath'].fillna(test_df['BsmtFullBath'].mode()[0])
test_df['BsmtHalfBath'] = test_df['BsmtHalfBath'].fillna(test_df['BsmtHalfBath'].mode()[0])
test_df['Functional'] = test_df['Functional'].fillna(test_df['Functional'].mode()[0])
test_df['FireplaceQu'] = test_df['FireplaceQu'].fillna(test_df['FireplaceQu'].mode()[0])
test_df['GarageType'] = test_df['GarageType'].fillna(test_df['GarageType'].mode()[0])
test_df['GarageFinish'] = test_df['GarageFinish'].fillna(test_df['GarageFinish'].mode()[0])
test_df['GarageQual'] = test_df['GarageQual'].fillna(test_df['GarageQual'].mode()[0])
test_df['GarageCond'] = test_df['GarageCond'].fillna(test_df['GarageCond'].mode()[0])
test_df.drop(['GarageYrBlt'], axis=1, inplace=True)
test_df.drop(['PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)
test_df.drop(['Id'], axis=1, inplace=True)
test_df.shape
test_df.loc[:, test_df.isnull().any()].head()
test_df['Exterior1st'] = test_df['Exterior1st'].fillna(test_df['Exterior1st'].mode()[0])
test_df['Exterior2nd'] = test_df['Exterior2nd'].fillna(test_df['Exterior2nd'].mode()[0])
test_df['BsmtFinSF1'] = test_df['BsmtFinSF1'].fillna(test_df['BsmtFinSF1'].mean())
test_df['BsmtFinSF2'] = test_df['BsmtFinSF2'].fillna(test_df['BsmtFinSF2'].mean())
test_df['BsmtUnfSF'] = test_df['BsmtUnfSF'].fillna(test_df['BsmtUnfSF'].mean())
test_df['TotalBsmtSF'] = test_df['TotalBsmtSF'].fillna(test_df['TotalBsmtSF'].mean())
test_df['KitchenQual'] = test_df['KitchenQual'].fillna(test_df['KitchenQual'].mode()[0])
test_df['GarageCars'] = test_df['GarageCars'].fillna(test_df['GarageCars'].mean())
test_df['GarageArea'] = test_df['GarageArea'].fillna(test_df['GarageArea'].mean())
test_df['SaleType'] = test_df['SaleType'].fillna(test_df['SaleType'].mode()[0])
test_df.loc[:, test_df.isnull().any()].head()
test_df.shape
df.shape
test_df.head()
final_df = pd.concat([df, test_df], axis=0)
final_df.shape
final_df = category_onehot_multcols(cat_feature)
final_df.shape
final_df = final_df.loc[:, ~final_df.columns.duplicated()]
final_df.shape
df_train = final_df.iloc[:1422, :]
df_test = final_df.iloc[1422:, :]
df_test.head()
df_train.head()
df_test.drop(['SalePrice'], axis=1, inplace=True)
df_train['SalePrice']
X_train = df_train.drop(['SalePrice'], axis=1)
y_train = df_train['SalePrice']
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
df_test = scalar.transform(df_test)
import xgboost
classi = xgboost.XGBRegressor(colsample_bytree=0.5, subsample=0.5, max_depth=4, min_child_weight=3.5, eta=0.05, n_estimators=2000, reg_alpha=0.5, reg_lambda=0.8, gamma=0.05)