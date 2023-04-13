import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
len(_input1.select_dtypes(include='object').columns)
_input1.head()
(_input1.shape, _input0.shape)
_input1.dtypes
na = _input1.isna().sum() / len(_input1)
na[na > 0.5]
_input1 = _input1.drop(columns=['PoolQC', 'Id'])
_input0 = _input0.drop(columns=['PoolQC', 'Id'])
_input1['Alley'] = _input1['Alley'].fillna('check')
print(_input1.groupby('Alley')['SalePrice'].mean())
_input1['Alley'] = _input1['Alley'].replace({'check': 0, 'Grvl': 1, 'Pave': 2})
_input0['Alley'] = _input0['Alley'].fillna('check')
_input0['Alley'] = _input0['Alley'].replace({'check': 0, 'Grvl': 1, 'Pave': 2})
_input1['Fence'] = _input1['Fence'].fillna('check')
print(_input1.groupby('Fence')['SalePrice'].mean())
_input1['Fence'] = _input1['Fence'].replace({'check': 0, 'MnPrv': 1, 'MnWw': 1, 'GdPrv': 2, 'GdWo': 2})
_input0['Fence'] = _input0['Fence'].fillna('check')
_input0['Fence'] = _input0['Fence'].replace({'check': 0, 'MnPrv': 1, 'MnWw': 1, 'GdPrv': 2, 'GdWo': 2})
_input1['MiscFeature'] = _input1['MiscFeature'].fillna('check')
print(_input1.groupby('MiscFeature')['SalePrice'].mean())
_input1['MiscFeature'] = _input1['MiscFeature'].replace({'check': 0, 'Othr': 1, 'Shed': 2, 'Gar2': 3, 'TenC': 4})
_input0['MiscFeature'] = _input0['MiscFeature'].fillna('check')
_input0['MiscFeature'] = _input0['MiscFeature'].replace({'check': 0, 'Othr': 1, 'Shed': 2, 'Gar2': 3, 'TenC': 4})
_input1['Street'] = _input1['Street'].fillna('check')
print(_input1.groupby('Street')['SalePrice'].mean())
_input1['Street'] = _input1['Street'].replace({'Grvl': 0, 'Pave': 1})
_input0['Street'] = _input0['Street'].fillna('check')
_input0['Street'] = _input0['Street'].replace({'Grvl': 0, 'Pave': 1})
_input1['Utilities'] = _input1['Utilities'].fillna('check')
print(_input1.groupby('Utilities')['SalePrice'].mean())
_input1['Utilities'] = _input1['Utilities'].replace({'NoSeWa': 0, 'AllPub': 1})
_input0['Utilities'] = _input0['Utilities'].fillna('check')
_input0['Utilities'] = _input0['Utilities'].replace({'NoSeWa': 0, 'AllPub': 1, 'check': 0})
_input1['CentralAir'] = _input1['CentralAir'].fillna('check')
print(_input1.groupby('CentralAir')['SalePrice'].mean())
_input1['CentralAir'] = _input1['CentralAir'].replace({'N': 0, 'Y': 1})
_input0['CentralAir'] = _input0['CentralAir'].replace({'N': 0, 'Y': 1})
_input1['LandSlope'] = _input1['LandSlope'].fillna('check')
print(_input1.groupby('LandSlope')['SalePrice'].mean())
_input1['LandSlope'] = _input1['LandSlope'].replace({'Gtl': 0, 'Mod': 1, 'Sev': 2})
_input0['LandSlope'] = _input0['LandSlope'].fillna('check')
_input0['LandSlope'] = _input0['LandSlope'].replace({'Gtl': 0, 'Mod': 1, 'Sev': 2})
_input1['GarageFinish'] = _input1['GarageFinish'].fillna('check')
print(_input1.groupby('GarageFinish')['SalePrice'].mean())
_input1['GarageFinish'] = _input1['GarageFinish'].replace({'check': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3})
_input0['GarageFinish'] = _input0['GarageFinish'].fillna('check')
_input0['GarageFinish'] = _input0['GarageFinish'].replace({'check': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3})
_input1['PavedDrive'] = _input1['PavedDrive'].fillna('check')
print(_input1.groupby('PavedDrive')['SalePrice'].mean())
_input1['PavedDrive'] = _input1['PavedDrive'].replace({'N': 0, 'P': 1, 'Y': 2})
_input0['PavedDrive'] = _input0['PavedDrive'].fillna('check')
_input0['PavedDrive'] = _input0['PavedDrive'].replace({'N': 0, 'P': 1, 'Y': 2})
_input1['LandContour'] = _input1['LandContour'].fillna('check')
print(_input1.groupby('LandContour')['SalePrice'].mean())
_input1['LandContour'] = _input1['LandContour'].replace({'Bnk': 0, 'Lvl': 1, 'Low': 2, 'HLS': 3})
_input0['LandContour'] = _input0['LandContour'].fillna('check')
_input0['LandContour'] = _input0['LandContour'].replace({'Bnk': 0, 'Lvl': 1, 'Low': 2, 'HLS': 3})
_input1['LotShape'] = _input1['LotShape'].fillna('check')
print(_input1.groupby('LotShape')['SalePrice'].mean())
_input1['LotShape'] = _input1['LotShape'].replace({'Reg': 0, 'IR1': 1, 'IR3': 2, 'IR2': 3})
_input0['LotShape'] = _input0['LotShape'].fillna('check')
_input0['LotShape'] = _input0['LotShape'].replace({'Reg': 0, 'IR1': 1, 'IR3': 2, 'IR2': 3})
_input1['ExterQual'] = _input1['ExterQual'].fillna('check')
print(_input1.groupby('ExterQual')['SalePrice'].mean())
_input1['ExterQual'] = _input1['ExterQual'].replace({'Fa': 0, 'TA': 1, 'Gd': 2, 'Ex': 3})
_input0['ExterQual'] = _input0['ExterQual'].fillna('check')
_input0['ExterQual'] = _input0['ExterQual'].replace({'Fa': 0, 'TA': 1, 'Gd': 2, 'Ex': 3})
_input1['MasVnrType'] = _input1['MasVnrType'].fillna('check')
print(_input1.groupby('MasVnrType')['SalePrice'].mean())
_input1['MasVnrType'] = _input1['MasVnrType'].replace({'BrkCmn': 0, 'None': 1, 'BrkFace': 2, 'check': 3, 'Stone': 4})
_input0['MasVnrType'] = _input0['MasVnrType'].fillna('check')
_input0['MasVnrType'] = _input0['MasVnrType'].replace({'BrkCmn': 0, 'None': 1, 'BrkFace': 2, 'check': 3, 'Stone': 4})
_input1['MSZoning'] = _input1['MSZoning'].fillna('check')
print(_input1.groupby('MSZoning')['SalePrice'].median())
_input1['MSZoning'] = _input1['MSZoning'].replace({'C (all)': 0, 'RM': 1, 'RH': 2, 'RL': 3, 'FV': 4})
_input0['MSZoning'] = _input0['MSZoning'].fillna('check')
_input0['MSZoning'] = _input0['MSZoning'].replace({'C (all)': 0, 'RM': 1, 'RH': 2, 'RL': 3, 'FV': 4, 'check': 2})
_input1['LotConfig'] = _input1['LotConfig'].fillna('check')
print(_input1.groupby('LotConfig')['SalePrice'].median())
_input1['LotConfig'] = _input1['LotConfig'].replace({'Corner': 0, 'FR2': 0, 'Inside': 0, 'CulDSac': 1, 'FR3': 1})
_input0['LotConfig'] = _input0['LotConfig'].fillna('check')
_input0['LotConfig'] = _input0['LotConfig'].replace({'Corner': 0, 'FR2': 0, 'Inside': 0, 'CulDSac': 1, 'FR3': 1})
_input1['BldgType'] = _input1['BldgType'].fillna('check')
print(_input1.groupby('BldgType')['SalePrice'].median())
_input1['BldgType'] = _input1['BldgType'].replace({'Twnhs': 0, '2fmCon': 0, 'Duplex': 0, '1Fam': 1, 'TwnhsE': 1})
_input0['BldgType'] = _input0['BldgType'].fillna('check')
_input0['BldgType'] = _input0['BldgType'].replace({'Twnhs': 0, '2fmCon': 0, 'Duplex': 0, '1Fam': 1, 'TwnhsE': 1})
_input1['BsmtQual'] = _input1['BsmtQual'].fillna('check')
print(_input1.groupby('BsmtQual')['SalePrice'].median())
_input1['BsmtQual'] = _input1['BsmtQual'].replace({'check': 0, 'Fa': 0, 'TA': 1, 'Gd': 2, 'Ex': 3})
_input0['BsmtQual'] = _input0['BsmtQual'].fillna('check')
_input0['BsmtQual'] = _input0['BsmtQual'].replace({'check': 0, 'Fa': 0, 'TA': 1, 'Gd': 2, 'Ex': 3})
_input1['RoofStyle'] = _input1['RoofStyle'].fillna('check')
print(_input1.groupby('RoofStyle')['SalePrice'].mean())
_input1['RoofStyle'] = _input1['RoofStyle'].replace({'Gambrel': 0, 'Gable': 1, 'Mansard': 1, 'Flat': 1, 'Shed': 2, 'Hip': 2})
_input0['RoofStyle'] = _input0['RoofStyle'].fillna('check')
_input0['RoofStyle'] = _input0['RoofStyle'].replace({'Gambrel': 0, 'Gable': 1, 'Mansard': 1, 'Flat': 1, 'Shed': 2, 'Hip': 2})
_input1['BsmtCond'] = _input1['BsmtCond'].fillna('check')
print(_input1.groupby('BsmtCond')['SalePrice'].mean())
_input1['BsmtCond'] = _input1['BsmtCond'].replace({'Po': 0, 'check': 1, 'Fa': 1, 'TA': 2, 'Gd': 2})
_input0['BsmtCond'] = _input0['BsmtCond'].fillna('check')
_input0['BsmtCond'] = _input0['BsmtCond'].replace({'Po': 0, 'check': 1, 'Fa': 1, 'TA': 2, 'Gd': 2})
_input1['BsmtExposure'] = _input1['BsmtExposure'].fillna('check')
print(_input1.groupby('BsmtExposure')['SalePrice'].mean())
_input1['BsmtExposure'] = _input1['BsmtExposure'].replace({'No': 1, 'check': 0, 'Mn': 1, 'Av': 1, 'Gd': 2})
_input0['BsmtExposure'] = _input0['BsmtExposure'].fillna('check')
_input0['BsmtExposure'] = _input0['BsmtExposure'].replace({'No': 1, 'check': 0, 'Mn': 1, 'Av': 1, 'Gd': 2})
_input1['KitchenQual'] = _input1['KitchenQual'].fillna('check')
print(_input1.groupby('KitchenQual')['SalePrice'].mean())
_input1['KitchenQual'] = _input1['KitchenQual'].replace({'Fa': 0, 'TA': 0, 'Gd': 1, 'Ex': 2})
_input0['KitchenQual'] = _input0['KitchenQual'].fillna('check')
_input0['KitchenQual'] = _input0['KitchenQual'].replace({'Fa': 0, 'TA': 0, 'Gd': 1, 'Ex': 2, 'check': 0})
_input1['ExterCond'] = _input1['ExterCond'].fillna('check')
print(_input1.groupby('ExterCond')['SalePrice'].mean())
_input1['ExterCond'] = _input1['ExterCond'].replace({'Fa': 0, 'Po': 0, 'Gd': 1, 'TA': 1, 'Ex': 2})
_input0['ExterCond'] = _input0['ExterCond'].fillna('check')
_input0['ExterCond'] = _input0['ExterCond'].replace({'Fa': 0, 'Po': 0, 'Gd': 1, 'TA': 1, 'Ex': 2})
_input1['Electrical'] = _input1['Electrical'].fillna('check')
print(_input1.groupby('Electrical')['SalePrice'].mean())
_input1['Electrical'] = _input1['Electrical'].replace({'FuseP': 0, 'Mix': 0, 'FuseA': 1, 'FuseF': 1, 'SBrkr': 2, 'check': 2})
_input0['Electrical'] = _input0['Electrical'].fillna('check')
_input0['Electrical'] = _input0['Electrical'].replace({'FuseP': 0, 'Mix': 0, 'FuseA': 1, 'FuseF': 1, 'SBrkr': 2, 'check': 2})
_input1['Heating'] = _input1['Heating'].fillna('check')
print(_input1.groupby('Heating')['SalePrice'].mean())
_input1['Heating'] = _input1['Heating'].replace({'Floor': 0, 'Grav': 0, 'Wall': 1, 'OthW': 2, 'GasW': 3, 'GasA': 4})
_input0['Heating'] = _input0['Heating'].fillna('check')
_input0['Heating'] = _input0['Heating'].replace({'Floor': 0, 'Grav': 0, 'Wall': 1, 'OthW': 2, 'GasW': 3, 'GasA': 4})
_input1['HeatingQC'] = _input1['HeatingQC'].fillna('check')
print(_input1.groupby('HeatingQC')['SalePrice'].mean())
_input1['HeatingQC'] = _input1['HeatingQC'].replace({'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 2, 'Ex': 3})
_input0['HeatingQC'] = _input0['HeatingQC'].fillna('check')
_input0['HeatingQC'] = _input0['HeatingQC'].replace({'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 2, 'Ex': 3})
_input1['FireplaceQu'] = _input1['FireplaceQu'].fillna('check')
print(_input1.groupby('FireplaceQu')['SalePrice'].mean())
_input1['FireplaceQu'] = _input1['FireplaceQu'].replace({'Po': 0, 'check': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
_input0['FireplaceQu'] = _input0['FireplaceQu'].fillna('check')
_input0['FireplaceQu'] = _input0['FireplaceQu'].replace({'Po': 0, 'check': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
_input1['GarageQual'] = _input1['GarageQual'].fillna('check')
print(_input1.groupby('GarageQual')['SalePrice'].mean())
_input1['GarageQual'] = _input1['GarageQual'].replace({'Po': 0, 'check': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})
_input0['GarageQual'] = _input0['GarageQual'].fillna('check')
_input0['GarageQual'] = _input0['GarageQual'].replace({'Po': 0, 'check': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})
_input1['GarageCond'] = _input1['GarageCond'].fillna('check')
print(_input1.groupby('GarageCond')['SalePrice'].mean())
_input1['GarageCond'] = _input1['GarageCond'].replace({'Po': 0, 'check': 0, 'Fa': 1, 'TA': 2, 'Gd': 2, 'Ex': 1})
_input0['GarageCond'] = _input0['GarageCond'].fillna('check')
_input0['GarageCond'] = _input0['GarageCond'].replace({'Po': 0, 'check': 0, 'Fa': 1, 'TA': 2, 'Gd': 2, 'Ex': 1})
_input1['Foundation'] = _input1['Foundation'].fillna('check')
print(_input1.groupby('Foundation')['SalePrice'].mean())
_input1['Foundation'] = _input1['Foundation'].replace({'Slab': 0, 'BrkTil': 1, 'CBlock': 2, 'Stone': 3, 'Wood': 4, 'PConc': 5})
_input0['Foundation'] = _input0['Foundation'].fillna('check')
_input0['Foundation'] = _input0['Foundation'].replace({'Slab': 0, 'BrkTil': 1, 'CBlock': 2, 'Stone': 3, 'Wood': 4, 'PConc': 5})
_input1['BsmtFinType1'] = _input1['BsmtFinType1'].fillna('check')
print(_input1.groupby('BsmtFinType1')['SalePrice'].mean())
_input1['BsmtFinType1'] = _input1['BsmtFinType1'].replace({'check': 0, 'Rec': 1, 'BLQ': 1, 'LwQ': 1, 'ALQ': 2, 'Unf': 2, 'GLQ': 3})
_input0['BsmtFinType1'] = _input0['BsmtFinType1'].fillna('check')
_input0['BsmtFinType1'] = _input0['BsmtFinType1'].replace({'check': 0, 'Rec': 1, 'BLQ': 1, 'LwQ': 1, 'ALQ': 2, 'Unf': 2, 'GLQ': 3})
_input1['BsmtFinType2'] = _input1['BsmtFinType2'].fillna('check')
print(_input1.groupby('BsmtFinType2')['SalePrice'].mean())
_input1['BsmtFinType2'] = _input1['BsmtFinType2'].replace({'check': 0, 'Rec': 1, 'BLQ': 1, 'LwQ': 1, 'ALQ': 3, 'Unf': 2, 'GLQ': 3})
_input0['BsmtFinType2'] = _input0['BsmtFinType2'].fillna('check')
_input0['BsmtFinType2'] = _input0['BsmtFinType2'].replace({'check': 0, 'Rec': 1, 'BLQ': 1, 'LwQ': 1, 'ALQ': 3, 'Unf': 2, 'GLQ': 2})
_input1['GarageType'] = _input1['GarageType'].fillna('check')
print(_input1.groupby('GarageType')['SalePrice'].mean())
_input1['GarageType'] = _input1['GarageType'].replace({'check': 0, 'CarPort': 0, 'Detchd': 1, '2Types': 2, 'Basment': 3, 'Attchd': 4, 'BuiltIn': 5})
_input0['GarageType'] = _input0['GarageType'].fillna('check')
_input0['GarageType'] = _input0['GarageType'].replace({'check': 0, 'CarPort': 0, 'Detchd': 1, '2Types': 2, 'Basment': 3, 'Attchd': 4, 'BuiltIn': 5})
_input1['SaleCondition'] = _input1['SaleCondition'].fillna('check')
print(_input1.groupby('SaleCondition')['SalePrice'].mean())
_input1['SaleCondition'] = _input1['SaleCondition'].replace({'AdjLand': 0, 'Abnorml': 1, 'Family': 1, 'Alloca': 2, 'Normal': 2, 'Partial': 3})
_input0['SaleCondition'] = _input0['SaleCondition'].fillna('check')
_input0['SaleCondition'] = _input0['SaleCondition'].replace({'AdjLand': 0, 'Abnorml': 1, 'Family': 1, 'Alloca': 2, 'Normal': 2, 'Partial': 3})
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
_input1[['Neighborhood', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Condition1', 'Condition2', 'HouseStyle', 'RoofMatl', 'Functional']] = _input1[['Neighborhood', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Condition1', 'Condition2', 'HouseStyle', 'RoofMatl', 'Functional']].fillna('aaaaaa')
_input1[['Neighborhood', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Condition1', 'Condition2', 'HouseStyle', 'RoofMatl', 'Functional']] = _input1[['Neighborhood', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Condition1', 'Condition2', 'HouseStyle', 'RoofMatl', 'Functional']].astype(str)
_input0[['Neighborhood', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Condition1', 'Condition2', 'HouseStyle', 'RoofMatl', 'Functional']] = _input0[['Neighborhood', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Condition1', 'Condition2', 'HouseStyle', 'RoofMatl', 'Functional']].fillna('aaaaaa')
_input0[['Neighborhood', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Condition1', 'Condition2', 'HouseStyle', 'RoofMatl', 'Functional']] = _input0[['Neighborhood', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Condition1', 'Condition2', 'HouseStyle', 'RoofMatl', 'Functional']].astype(str)
_input1['Neighborhood'] = label_encoder.fit_transform(_input1['Neighborhood'])
_input1['Exterior1st'] = label_encoder.fit_transform(_input1['Exterior1st'])
_input1['Exterior2nd'] = label_encoder.fit_transform(_input1['Exterior2nd'])
_input1['SaleType'] = label_encoder.fit_transform(_input1['SaleType'])
_input1['Condition1'] = label_encoder.fit_transform(_input1['Condition1'])
_input1['Condition2'] = label_encoder.fit_transform(_input1['Condition2'])
_input1['HouseStyle'] = label_encoder.fit_transform(_input1['HouseStyle'])
_input1['RoofMatl'] = label_encoder.fit_transform(_input1['RoofMatl'])
_input1['Functional'] = label_encoder.fit_transform(_input1['Functional'])
_input0['Neighborhood'] = label_encoder.fit_transform(_input0['Neighborhood'])
_input0['Exterior1st'] = label_encoder.fit_transform(_input0['Exterior1st'])
_input0['Exterior2nd'] = label_encoder.fit_transform(_input0['Exterior2nd'])
_input0['SaleType'] = label_encoder.fit_transform(_input0['SaleType'])
_input0['Condition1'] = label_encoder.fit_transform(_input0['Condition1'])
_input0['Condition2'] = label_encoder.fit_transform(_input0['Condition2'])
_input0['HouseStyle'] = label_encoder.fit_transform(_input0['HouseStyle'])
_input0['RoofMatl'] = label_encoder.fit_transform(_input0['RoofMatl'])
_input0['Functional'] = label_encoder.fit_transform(_input0['Functional'])
for i in _input1.select_dtypes(include='object').columns:
    print('____________________________________________________')
    print(i, _input1[i].nunique())
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
model = CatBoostRegressor()
X = _input1.drop(columns=['SalePrice'])
y = _input1['SalePrice']
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=42)