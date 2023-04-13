import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1
_input1.isnull().sum()
_input1.info()
sns.heatmap(_input1.isnull(), cbar=False)
sns.heatmap(_input1.corr())
data = _input1.drop(['PoolQC', 'Fence', 'MiscFeature', 'Alley', 'FireplaceQu'], axis=1)
data
data['SaleType'] = data['SaleType'].fillna(data['SaleType'].mode()[0], inplace=False)
data['MSZoning'] = data['MSZoning'].fillna(data['MSZoning'].mode()[0], inplace=False)
data['LotFrontage'] = data['LotFrontage'].fillna(data['LotFrontage'].mode()[0], inplace=False)
data['Utilities'] = data['Utilities'].fillna(data['Utilities'].mode()[0], inplace=False)
data['Exterior1st'] = data['Exterior1st'].fillna(data['Exterior1st'].mode()[0], inplace=False)
data['Exterior2nd'] = data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0], inplace=False)
data['MasVnrType'] = data['MasVnrType'].fillna(data['MasVnrType'].mode()[0], inplace=False)
data['MasVnrArea'] = data['MasVnrArea'].fillna(data['MasVnrArea'].mode()[0], inplace=False)
data['BsmtQual'] = data['BsmtQual'].fillna(data['BsmtQual'].mode()[0], inplace=False)
data['BsmtCond'] = data['BsmtCond'].fillna(data['BsmtCond'].mode()[0], inplace=False)
data['BsmtExposure'] = data['BsmtExposure'].fillna(data['BsmtExposure'].mode()[0], inplace=False)
data['BsmtFinType1'] = data['BsmtFinType1'].fillna(data['BsmtFinType1'].mode()[0], inplace=False)
data['BsmtFinSF1'] = data['BsmtFinSF1'].fillna(data['BsmtFinSF1'].mode()[0], inplace=False)
data['BsmtFinType2'] = data['BsmtFinType2'].fillna(data['BsmtFinType2'].mode()[0], inplace=False)
data['BsmtFinSF2'] = data['BsmtFinSF2'].fillna(data['BsmtFinSF2'].mode()[0], inplace=False)
data['BsmtUnfSF'] = data['BsmtUnfSF'].fillna(data['BsmtUnfSF'].mode()[0], inplace=False)
data['TotalBsmtSF'] = data['TotalBsmtSF'].fillna(data['TotalBsmtSF'].mode()[0], inplace=False)
data['BsmtFullBath'] = data['BsmtFullBath'].fillna(data['BsmtFullBath'].mode()[0], inplace=False)
data['BsmtFullBath'] = data['BsmtFullBath'].fillna(data['BsmtFullBath'].mode()[0], inplace=False)
data['KitchenQual'] = data['KitchenQual'].fillna(data['KitchenQual'].mode()[0], inplace=False)
data['Functional'] = data['Functional'].fillna(data['Functional'].mode()[0], inplace=False)
data['GarageType'] = data['GarageType'].fillna(data['GarageType'].mode()[0], inplace=False)
data['GarageYrBlt'] = data['GarageYrBlt'].fillna(data['GarageYrBlt'].mode()[0], inplace=False)
data['GarageFinish'] = data['GarageFinish'].fillna(data['GarageFinish'].mode()[0], inplace=False)
data['GarageCars'] = data['GarageCars'].fillna(data['GarageCars'].mode()[0], inplace=False)
data['GarageArea'] = data['GarageArea'].fillna(data['GarageArea'].mode()[0], inplace=False)
data['GarageQual'] = data['GarageQual'].fillna(data['GarageQual'].mode()[0], inplace=False)
data['Functional'] = data['Functional'].fillna(data['Functional'].mode()[0], inplace=False)
data['GarageType'] = data['GarageType'].fillna(data['GarageType'].mode()[0], inplace=False)
data['GarageYrBlt'] = data['GarageYrBlt'].fillna(data['GarageYrBlt'].mode()[0], inplace=False)
data['GarageFinish'] = data['GarageFinish'].fillna(data['GarageFinish'].mode()[0], inplace=False)
data['GarageCars'] = data['GarageCars'].fillna(data['GarageCars'].mode()[0], inplace=False)
data['GarageArea'] = data['GarageArea'].fillna(data['GarageArea'].mode()[0], inplace=False)
data['GarageQual'] = data['GarageQual'].fillna(data['GarageQual'].mode()[0], inplace=False)
data['GarageCond'] = data['GarageCond'].fillna(data['GarageCond'].mode()[0], inplace=False)
data['SaleType'] = data['SaleType'].fillna(data['SaleType'].mode()[0], inplace=False)
data['Electrical'] = data['Electrical'].fillna(data['Electrical'].mode()[0], inplace=False)
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
y = y.fillna(y.mean(), inplace=False)
from sklearn.preprocessing import StandardScaler