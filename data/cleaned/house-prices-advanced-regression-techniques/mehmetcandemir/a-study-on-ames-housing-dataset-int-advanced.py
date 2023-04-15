import pandas as pd
import numpy as np
import time
import itertools as it
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df_TRAIN = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df_TEST = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
df = pd.concat((df_TRAIN, df_TEST), ignore_index=True)
df.drop('Id', axis=1, inplace=True)
df['Alley'] = df['Alley'].fillna('None')
df['MasVnrType'] = df['MasVnrType'].fillna('None')
df['BsmtQual'] = df['BsmtQual'].fillna('NA')
df['BsmtCond'] = df['BsmtCond'].fillna('NA')
df['BsmtExposure'] = df['BsmtExposure'].fillna('NA')
df['BsmtFinType1'] = df['BsmtFinType1'].fillna('NA')
df['BsmtFinType2'] = df['BsmtFinType2'].fillna('NA')
df['FireplaceQu'] = df['FireplaceQu'].fillna('NA')
df['GarageType'] = df['GarageType'].fillna('NA')
df['GarageFinish'] = df['GarageFinish'].fillna('NA')
df['GarageQual'] = df['GarageQual'].fillna('NA')
df['GarageCond'] = df['GarageCond'].fillna('NA')
df['PoolQC'] = df['PoolQC'].fillna('NA')
df['Fence'] = df['Fence'].fillna('NA')
df['MiscFeature'] = df['MiscFeature'].fillna('NA')
df['SaleType'] = df['SaleType'].fillna('Oth')
df['LotFrontage'] = df['LotFrontage'].fillna(0)
df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
df['TotalBsmtSF'] = df['TotalBsmtSF'].fillna(0)
df['BsmtFinSF1'] = df['BsmtFinSF1'].fillna(0)
df['BsmtFinSF2'] = df['BsmtFinSF2'].fillna(0)
df['BsmtUnfSF'] = df['BsmtUnfSF'].fillna(0)
df['BsmtFullBath'] = df['BsmtFullBath'].fillna(0)
df['BsmtHalfBath'] = df['BsmtHalfBath'].fillna(0)
df['GarageCars'] = df['GarageCars'].fillna(0)
df['GarageArea'] = df['GarageArea'].fillna(0)
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['GarageYrBlt'].mean())
df['MSSubClass'] = df['MSSubClass'].replace({20: 'SC20', 30: 'SC30', 40: 'SC40', 45: 'SC45', 50: 'SC50', 60: 'SC60', 70: 'SC70', 75: 'SC75', 80: 'SC80', 85: 'SC85', 90: 'SC90', 120: 'SC120', 150: 'SC150', 160: 'SC160', 180: 'SC180', 190: 'SC190'})
df['MoSold'] = df['MoSold'].replace({1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MAY', 6: 'JUN', 7: 'JUL', 8: 'AUG', 9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'})
df['OverallQual'] = df['OverallQual'].replace({(1, 2, 3): 1, (4, 5, 6): 2, (7, 8, 9, 10): 3})
df['OverallCond'] = df['OverallCond'].replace({(1, 2, 3): 1, (4, 5, 6): 2, (7, 8, 9, 10): 3})
df['ExterQual'] = df['ExterQual'].replace({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
df['ExterCond'] = df['ExterCond'].replace({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
df['BsmtQual'] = df['BsmtQual'].replace({'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
df['BsmtCond'] = df['BsmtCond'].replace({'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
df['BsmtExposure'] = df['BsmtExposure'].replace({'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
df['BsmtFinType1'] = df['BsmtFinType1'].replace({'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6})
df['BsmtFinType2'] = df['BsmtFinType2'].replace({'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6})
df['HeatingQC'] = df['HeatingQC'].replace({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
df['KitchenQual'] = df['KitchenQual'].replace({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
df['Functional'] = df['Functional'].replace({'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8})
df['FireplaceQu'] = df['FireplaceQu'].replace({'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
df['GarageQual'] = df['GarageQual'].replace({'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
df['GarageCond'] = df['GarageCond'].replace({'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
df['PoolQC'] = df['PoolQC'].replace({'NA': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})
df['Fence'] = df['Fence'].replace({'NA': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4})
df['ExterScore'] = df['ExterQual'] * df['ExterCond']
df['BsmtScore'] = df['TotalBsmtSF'] * df['BsmtQual'] * df['BsmtCond']
df['FireplaceScore'] = df['FireplaceQu'] * df['Fireplaces']
df['GarageScore'] = df['GarageQual'] * df['GarageCond'] * df['GarageArea'] * df['GarageCars']
df['PoolScore'] = df['PoolArea'] * df['PoolQC']
df.corr()['SalePrice'].sort_values(ascending=False)
features = df.corr()[df.corr()['SalePrice'] > 0.3]['SalePrice'].sort_values(ascending=False).index
df_TRAIN = df[:df_TRAIN.shape[0]].copy()
df_TEST = df[df_TRAIN.shape[0]:].copy()
for col in df.loc[:, df.columns != 'SalePrice'].columns[:]:
    if df_TEST[col].isna().sum():
        print(f'TEST - {col}, {df[col].isna().sum()}')
    if df_TRAIN[col].isna().sum():
        print(f'TRAIN - {col}, {df[col].isna().sum()}')
for col in df.loc[:, df.columns != 'SalePrice'].columns:
    if df_TEST[col].isna().sum():
        if df_TEST[col].dtype != 'object':
            df_TEST[col] = df_TEST[col].fillna(df_TEST[col].mean())
        else:
            df_TEST[col] = df_TEST[col].fillna(df_TEST[col].mode()[0])
    if df_TRAIN[col].isna().sum():
        if df_TRAIN[col].dtype != 'object':
            df_TRAIN[col] = df_TRAIN[col].fillna(df_TRAIN[col].mean())
        else:
            df_TRAIN[col] = df_TRAIN[col].fillna(df_TRAIN[col].mode()[0])
(fig, ax) = plt.subplots(4, 4, figsize=(18, 16))
iters = [iter(features[1:17])] * 4
iter_features = list(zip(*iters))
for i in range(4):
    for j in range(4):
        ax[i][j].scatter(x=df_TRAIN[iter_features[i][j]], y=df_TRAIN['SalePrice'], label=iter_features[i][j], c=df_TRAIN[iter_features[i][j]], cmap='viridis')
        ax[i][j].set_xlabel(iter_features[i][j])
        ax[i][j].set_ylabel('Sale Price')
plt.tight_layout()
df_TRAIN.drop(df_TRAIN[df_TRAIN['BsmtScore'] > 60000].index, axis=0, inplace=True)
df_TRAIN.drop(df_TRAIN[df_TRAIN['TotalBsmtSF'] > 6000].index, axis=0, inplace=True)
df_TRAIN.drop(df_TRAIN[df_TRAIN['GrLivArea'] > 4000].index, axis=0, inplace=True)
df_TRAIN.drop(df_TRAIN[df_TRAIN['1stFlrSF'] > 4000].index, axis=0, inplace=True)
df_TRAIN.drop(df_TRAIN[df_TRAIN['MasVnrArea'] > 1250].index, axis=0, inplace=True)
df = pd.concat((df_TRAIN, df_TEST))
for col in df[features[1:17]].columns[:10]:
    df[col + '_**2'] = df[col] ** 2
    df[col + '_**3'] = df[col] ** 3
    df[col + '_**sqr'] = np.sqrt(df[col])
df[df.select_dtypes(exclude='object').columns] = df[df.select_dtypes(exclude='object').columns].apply(np.log1p)
df = pd.get_dummies(df)
X_TRAIN = df.drop('SalePrice', axis=1).to_numpy()[:df_TRAIN.shape[0]]
y_TRAIN = df['SalePrice'].to_numpy()[:df_TRAIN.shape[0]]
X_TEST = df.drop('SalePrice', axis=1).to_numpy()[df_TRAIN.shape[0]:]
print(f'X TRAIN: {X_TRAIN.shape}, y TRAIN: {y_TRAIN.shape}')
print(f'X TEST: {X_TEST.shape}')
(X_train, X_test, y_train, y_test) = train_test_split(X_TRAIN, y_TRAIN, random_state=42, shuffle=True)
print(f'X train: {X_train.shape}, y train: {y_train.shape}')
print(f'X test: {X_test.shape}, y test: {y_test.shape}')

def gridsearch(model, parameters):
    grid = GridSearchCV(estimator=model, param_grid=parameters, cv=2, n_jobs=-1)