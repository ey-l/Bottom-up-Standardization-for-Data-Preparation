import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df
df.isnull().sum()
df.info()
sns.heatmap(df.isnull(), cbar=False)
sns.heatmap(df.corr())
data = df.drop(['PoolQC', 'Fence', 'MiscFeature', 'Alley', 'FireplaceQu'], axis=1)
data
data['SaleType'].fillna(data['SaleType'].mode()[0], inplace=True)
data['MSZoning'].fillna(data['MSZoning'].mode()[0], inplace=True)
data['LotFrontage'].fillna(data['LotFrontage'].mode()[0], inplace=True)
data['Utilities'].fillna(data['Utilities'].mode()[0], inplace=True)
data['Exterior1st'].fillna(data['Exterior1st'].mode()[0], inplace=True)
data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0], inplace=True)
data['MasVnrType'].fillna(data['MasVnrType'].mode()[0], inplace=True)
data['MasVnrArea'].fillna(data['MasVnrArea'].mode()[0], inplace=True)
data['BsmtQual'].fillna(data['BsmtQual'].mode()[0], inplace=True)
data['BsmtCond'].fillna(data['BsmtCond'].mode()[0], inplace=True)
data['BsmtExposure'].fillna(data['BsmtExposure'].mode()[0], inplace=True)
data['BsmtFinType1'].fillna(data['BsmtFinType1'].mode()[0], inplace=True)
data['BsmtFinSF1'].fillna(data['BsmtFinSF1'].mode()[0], inplace=True)
data['BsmtFinType2'].fillna(data['BsmtFinType2'].mode()[0], inplace=True)
data['BsmtFinSF2'].fillna(data['BsmtFinSF2'].mode()[0], inplace=True)
data['BsmtUnfSF'].fillna(data['BsmtUnfSF'].mode()[0], inplace=True)
data['TotalBsmtSF'].fillna(data['TotalBsmtSF'].mode()[0], inplace=True)
data['BsmtFullBath'].fillna(data['BsmtFullBath'].mode()[0], inplace=True)
data['BsmtFullBath'].fillna(data['BsmtFullBath'].mode()[0], inplace=True)
data['KitchenQual'].fillna(data['KitchenQual'].mode()[0], inplace=True)
data['Functional'].fillna(data['Functional'].mode()[0], inplace=True)
data['GarageType'].fillna(data['GarageType'].mode()[0], inplace=True)
data['GarageYrBlt'].fillna(data['GarageYrBlt'].mode()[0], inplace=True)
data['GarageFinish'].fillna(data['GarageFinish'].mode()[0], inplace=True)
data['GarageCars'].fillna(data['GarageCars'].mode()[0], inplace=True)
data['GarageArea'].fillna(data['GarageArea'].mode()[0], inplace=True)
data['GarageQual'].fillna(data['GarageQual'].mode()[0], inplace=True)
data['Functional'].fillna(data['Functional'].mode()[0], inplace=True)
data['GarageType'].fillna(data['GarageType'].mode()[0], inplace=True)
data['GarageYrBlt'].fillna(data['GarageYrBlt'].mode()[0], inplace=True)
data['GarageFinish'].fillna(data['GarageFinish'].mode()[0], inplace=True)
data['GarageCars'].fillna(data['GarageCars'].mode()[0], inplace=True)
data['GarageArea'].fillna(data['GarageArea'].mode()[0], inplace=True)
data['GarageQual'].fillna(data['GarageQual'].mode()[0], inplace=True)
data['GarageCond'].fillna(data['GarageCond'].mode()[0], inplace=True)
data['SaleType'].fillna(data['SaleType'].mode()[0], inplace=True)
data['Electrical'].fillna(data['Electrical'].mode()[0], inplace=True)
data.info()
data.isnull().sum()
sns.heatmap(data.isnull(), cbar=False)
dt = data
from sklearn.preprocessing import LabelEncoder
labelencoder_ = LabelEncoder()
labelencoder_tst = LabelEncoder()
data.iloc[:, 13].dtype
for i in range(0, 76):
    if data.iloc[:, i].dtype == 'O':
        dt.iloc[:, i] = labelencoder_.fit_transform(data.iloc[:, i])
X = dt.drop(['SalePrice', 'Id'], axis=1)
y = dt['SalePrice']
y.fillna(y.mean(), inplace=True)
from sklearn.preprocessing import StandardScaler