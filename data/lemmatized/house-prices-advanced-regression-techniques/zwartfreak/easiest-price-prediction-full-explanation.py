import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
(_input1.shape, _input0.shape)
_input1.head()
_input0.head()
_input1.count()
_input1.dtypes
_input1.describe(include='all')
_input1.info
_input1.isnull()
_input1.isnull().sum().sum()
_input0.isnull().sum().sum()
_input1.isnull().sum()
y = _input1.SalePrice
X = _input1.drop(columns=['SalePrice'], axis=1)
(y.shape, X.shape, _input0.shape)
X['Type'] = 'train'
_input0['Type'] = 'test'
data = X.append(_input0)
data.isnull().sum().sum()
columns_having_null_values = data[data.columns[data.isnull().sum() > 0]]
columns_having_null_values
data['Electrical'].value_counts()
data['Electrical'] = data['Electrical'].fillna('Sbrkr', inplace=False)
data['MSZoning'].value_counts()
data['MSZoning'] = data['MSZoning'].fillna('RL', inplace=False)
data['LotFrontage'] = data['LotFrontage'].fillna(data['LotFrontage'].mean(), inplace=False)
data['Alley'] = data['Alley'].fillna('Nothing', inplace=False)
data['Utilities'] = data['Utilities'].fillna('AllPub', inplace=False)
data['Exterior1st'] = data['Exterior1st'].fillna('VinylSd', inplace=False)
data['Exterior2nd'] = data['Exterior2nd'].fillna('VinylSd', inplace=False)
data['MasVnrArea'] = data['MasVnrArea'].fillna(0, inplace=False)
data['MasVnrType'] = data['MasVnrType'].fillna('None', inplace=False)
data['BsmtCond'] = data['BsmtCond'].fillna('No', inplace=False)
data['BsmtExposure'] = data['BsmtExposure'].fillna('NB', inplace=False)
data['BsmtFinType1'] = data['BsmtFinType1'].fillna('NB', inplace=False)
data['BsmtFinSF1'] = data['BsmtFinSF1'].fillna(0.0, inplace=False)
data['BsmtFinSF2'] = data['BsmtFinSF2'].fillna(0.0, inplace=False)
data['BsmtUnfSF'] = data['BsmtUnfSF'].fillna(0.0, inplace=False)
data['TotalBsmtSF'] = data['TotalBsmtSF'].fillna(0.0, inplace=False)
data['BsmtFullBath'] = data['BsmtFullBath'].fillna(0.0, inplace=False)
data['BsmtHalfBath'] = data['BsmtHalfBath'].fillna(0.0, inplace=False)
data['KitchenQual'] = data['KitchenQual'].fillna('TA', inplace=False)
data['Functional'] = data['Functional'].fillna('Typ', inplace=False)
data['FireplaceQu'] = data['FireplaceQu'].fillna('None', inplace=False)
data['GarageType'] = data['GarageType'].fillna('No', inplace=False)
data['GarageYrBlt'] = data['GarageYrBlt'].fillna(0, inplace=False)
data['GarageFinish'] = data['GarageFinish'].fillna('No', inplace=False)
data['GarageCars'] = data['GarageCars'].fillna(0, inplace=False)
data['GarageArea'] = data['GarageArea'].fillna(0, inplace=False)
data['GarageQual'] = data['GarageQual'].fillna('No', inplace=False)
data['GarageCond'] = data['GarageCond'].fillna('No', inplace=False)
data['PoolQC'] = data['PoolQC'].fillna('No', inplace=False)
data['Fence'] = data['Fence'].fillna('No', inplace=False)
data['MiscFeature'] = data['MiscFeature'].fillna('No', inplace=False)
data['SaleType'] = data['SaleType'].fillna('Con', inplace=False)
data['SaleCondition'] = data['SaleCondition'].fillna('None', inplace=False)
data['BsmtQual'] = data['BsmtQual'].fillna('TA', inplace=False)
data['BsmtFinType2'] = data['BsmtFinType2'].fillna('Unf', inplace=False)
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
data = data.drop(data[drop_columns], axis=1, inplace=False)
data.head()
from sklearn.preprocessing import LabelEncoder
for i in object_columnns:
    label = LabelEncoder()