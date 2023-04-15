import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
(train.shape, test.shape)
train.head()
test.head()
train.count()
train.dtypes
train.describe(include='all')
train.info
train.isnull()
train.isnull().sum().sum()
test.isnull().sum().sum()
train.isnull().sum()
y = train.SalePrice
X = train.drop(columns=['SalePrice'], axis=1)
(y.shape, X.shape, test.shape)
X['Type'] = 'train'
test['Type'] = 'test'
data = X.append(test)
data.isnull().sum().sum()
columns_having_null_values = data[data.columns[data.isnull().sum() > 0]]
columns_having_null_values
data['Electrical'].value_counts()
data['Electrical'].fillna('Sbrkr', inplace=True)
data['MSZoning'].value_counts()
data['MSZoning'].fillna('RL', inplace=True)
data['LotFrontage'].fillna(data['LotFrontage'].mean(), inplace=True)
data['Alley'].fillna('Nothing', inplace=True)
data['Utilities'].fillna('AllPub', inplace=True)
data['Exterior1st'].fillna('VinylSd', inplace=True)
data['Exterior2nd'].fillna('VinylSd', inplace=True)
data['MasVnrArea'].fillna(0, inplace=True)
data['MasVnrType'].fillna('None', inplace=True)
data['BsmtCond'].fillna('No', inplace=True)
data['BsmtExposure'].fillna('NB', inplace=True)
data['BsmtFinType1'].fillna('NB', inplace=True)
data['BsmtFinSF1'].fillna(0.0, inplace=True)
data['BsmtFinSF2'].fillna(0.0, inplace=True)
data['BsmtUnfSF'].fillna(0.0, inplace=True)
data['TotalBsmtSF'].fillna(0.0, inplace=True)
data['BsmtFullBath'].fillna(0.0, inplace=True)
data['BsmtHalfBath'].fillna(0.0, inplace=True)
data['KitchenQual'].fillna('TA', inplace=True)
data['Functional'].fillna('Typ', inplace=True)
data['FireplaceQu'].fillna('None', inplace=True)
data['GarageType'].fillna('No', inplace=True)
data['GarageYrBlt'].fillna(0, inplace=True)
data['GarageFinish'].fillna('No', inplace=True)
data['GarageCars'].fillna(0, inplace=True)
data['GarageArea'].fillna(0, inplace=True)
data['GarageQual'].fillna('No', inplace=True)
data['GarageCond'].fillna('No', inplace=True)
data['PoolQC'].fillna('No', inplace=True)
data['Fence'].fillna('No', inplace=True)
data['MiscFeature'].fillna('No', inplace=True)
data['SaleType'].fillna('Con', inplace=True)
data['SaleCondition'].fillna('None', inplace=True)
data['BsmtQual'].fillna('TA', inplace=True)
data['BsmtFinType2'].fillna('Unf', inplace=True)
data.isnull().sum().sum()
int_columns = data[data.columns[data.dtypes == 'int']]
int_columns.columns
data['MSZoning'].unique()
object_columnns = data[data.columns[data.dtypes == 'object']]
object_columnns.columns
float_columns = data[data.columns[data.dtypes == 'float']]
float_columns.columns
data.var()
corr_matrix = data.corr()
corr_matrix
upper_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
upper_matrix
drop_columns = [col for col in upper_matrix.columns if any(upper_matrix[col] > 0.85)]
drop_columns
data.drop(data[drop_columns], axis=1, inplace=True)
data.head()
from sklearn.preprocessing import LabelEncoder
for i in object_columnns:
    label = LabelEncoder()