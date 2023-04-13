import numpy as np
import pandas as pd
import datetime
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
import lightgbm as lgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
pd.set_option('display.max_columns', None)
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1 = _input1.drop(_input1[(_input1['OverallQual'] < 5) & (_input1['SalePrice'] > 200000)].index, inplace=False)
_input1 = _input1.drop(_input1[(_input1['GrLivArea'] > 4000) & (_input1['SalePrice'] < 300000)].index, inplace=False)
_input1 = _input1.reset_index(drop=True, inplace=False)
_input1['MSSubClass'] = _input1['MSSubClass'].apply(str)
_input1['YrSold'] = _input1['YrSold'].astype(str)
_input1['MoSold'] = _input1['MoSold'].astype(str)

def fill_missings(res):
    res['Alley'] = res['Alley'].fillna('missing')
    res['PoolQC'] = res['PoolQC'].fillna(res['PoolQC'].mode()[0])
    res['MasVnrType'] = res['MasVnrType'].fillna('None')
    res['BsmtQual'] = res['BsmtQual'].fillna(res['BsmtQual'].mode()[0])
    res['BsmtCond'] = res['BsmtCond'].fillna(res['BsmtCond'].mode()[0])
    res['FireplaceQu'] = res['FireplaceQu'].fillna(res['FireplaceQu'].mode()[0])
    res['GarageType'] = res['GarageType'].fillna('missing')
    res['GarageFinish'] = res['GarageFinish'].fillna(res['GarageFinish'].mode()[0])
    res['GarageQual'] = res['GarageQual'].fillna(res['GarageQual'].mode()[0])
    res['GarageCond'] = res['GarageCond'].fillna('missing')
    res['Fence'] = res['Fence'].fillna('missing')
    res['Street'] = res['Street'].fillna('missing')
    res['LotShape'] = res['LotShape'].fillna('missing')
    res['LandContour'] = res['LandContour'].fillna('missing')
    res['BsmtExposure'] = res['BsmtExposure'].fillna(res['BsmtExposure'].mode()[0])
    res['BsmtFinType1'] = res['BsmtFinType1'].fillna('missing')
    res['BsmtFinType2'] = res['BsmtFinType2'].fillna('missing')
    res['CentralAir'] = res['CentralAir'].fillna('missing')
    res['Electrical'] = res['Electrical'].fillna(res['Electrical'].mode()[0])
    res['MiscFeature'] = res['MiscFeature'].fillna('missing')
    res['MSZoning'] = res['MSZoning'].fillna(res['MSZoning'].mode()[0])
    res['Utilities'] = res['Utilities'].fillna('missing')
    res['Exterior1st'] = res['Exterior1st'].fillna(res['Exterior1st'].mode()[0])
    res['Exterior2nd'] = res['Exterior2nd'].fillna(res['Exterior2nd'].mode()[0])
    res['KitchenQual'] = res['KitchenQual'].fillna(res['KitchenQual'].mode()[0])
    res['Functional'] = res['Functional'].fillna('Typ')
    res['SaleType'] = res['SaleType'].fillna(res['SaleType'].mode()[0])
    res['SaleCondition'] = res['SaleCondition'].fillna('missing')
    flist = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
    for fl in flist:
        res[fl] = res[fl].fillna(0)
    res['TotalBsmtSF'] = res['TotalBsmtSF'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
    res['2ndFlrSF'] = res['2ndFlrSF'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)
    res['GarageArea'] = res['GarageArea'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
    res['GarageCars'] = res['GarageCars'].apply(lambda x: 0 if x <= 0.0 else x)
    res['LotFrontage'] = res['LotFrontage'].apply(lambda x: np.exp(4.2) if x <= 0.0 else x)
    res['MasVnrArea'] = res['MasVnrArea'].apply(lambda x: np.exp(4) if x <= 0.0 else x)
    res['BsmtFinSF1'] = res['BsmtFinSF1'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)
    return res
_input1 = fill_missings(_input1)
_input1['TotalSF'] = _input1['TotalBsmtSF'] + _input1['1stFlrSF'] + _input1['2ndFlrSF']

def QualToInt(x):
    if x == 'Ex':
        r = 0
    elif x == 'Gd':
        r = 1
    elif x == 'TA':
        r = 2
    elif x == 'Fa':
        r = 3
    elif x == 'missing':
        r = 4
    else:
        r = 5
    return r
_input1['ExterQual'] = _input1['ExterQual'].apply(QualToInt)
_input1['ExterCond'] = _input1['ExterCond'].apply(QualToInt)
_input1['KitchenQual'] = _input1['KitchenQual'].apply(QualToInt)
_input1['HeatingQC'] = _input1['HeatingQC'].apply(QualToInt)
_input1['BsmtQual'] = _input1['BsmtQual'].apply(QualToInt)
_input1['BsmtCond'] = _input1['BsmtCond'].apply(QualToInt)
_input1['FireplaceQu'] = _input1['FireplaceQu'].apply(QualToInt)
_input1['GarageQual'] = _input1['GarageQual'].apply(QualToInt)
_input1['PoolQC'] = _input1['PoolQC'].apply(QualToInt)

def SlopeToInt(x):
    if x == 'Gtl':
        r = 0
    elif x == 'Mod':
        r = 1
    elif x == 'Sev':
        r = 2
    else:
        r = 3
    return r
_input1['LandSlope'] = _input1['LandSlope'].apply(SlopeToInt)
_input1['CentralAir'] = _input1['CentralAir'].apply(lambda x: 0 if x == 'N' else 1)
_input1['Street'] = _input1['Street'].apply(lambda x: 0 if x == 'Pave' else 1)
_input1['PavedDrive'] = _input1['PavedDrive'].apply(lambda x: 0 if x == 'Y' else 1)

def GFinishToInt(x):
    if x == 'Fin':
        r = 0
    elif x == 'RFn':
        r = 1
    elif x == 'Unf':
        r = 2
    else:
        r = 3
    return r
_input1['GarageFinish'] = _input1['GarageFinish'].apply(GFinishToInt)

def BsmtExposureToInt(x):
    if x == 'Gd':
        r = 0
    elif x == 'Av':
        r = 1
    elif x == 'Mn':
        r = 2
    elif x == 'No':
        r = 3
    else:
        r = 4
    return r
_input1['BsmtExposure'] = _input1['BsmtExposure'].apply(BsmtExposureToInt)

def FunctionalToInt(x):
    if x == 'Typ':
        r = 0
    elif x == 'Min1':
        r = 1
    elif x == 'Min2':
        r = 1
    else:
        r = 2
    return r
_input1['Functional_int'] = _input1['Functional'].apply(FunctionalToInt)

def HouseStyleToInt(x):
    if x == '1.5Unf':
        r = 0
    elif x == 'SFoyer':
        r = 1
    elif x == '1.5Fin':
        r = 2
    elif x == '2.5Unf':
        r = 3
    elif x == 'SLvl':
        r = 4
    elif x == '1Story':
        r = 5
    elif x == '2Story':
        r = 6
    elif x == ' 2.5Fin':
        r = 7
    else:
        r = 8
    return r
_input1['HouseStyle_int'] = _input1['HouseStyle'].apply(HouseStyleToInt)
_input1['HouseStyle_1st'] = 1 * (_input1['HouseStyle'] == '1Story')
_input1['HouseStyle_2st'] = 1 * (_input1['HouseStyle'] == '2Story')
_input1['HouseStyle_15st'] = 1 * (_input1['HouseStyle'] == '1.5Fin')

def FoundationToInt(x):
    if x == 'PConc':
        r = 3
    elif x == 'CBlock':
        r = 2
    elif x == 'BrkTil':
        r = 1
    else:
        r = 0
    return r
_input1['Foundation_int'] = _input1['Foundation'].apply(FoundationToInt)

def MasVnrTypeToInt(x):
    if x == 'Stone':
        r = 3
    elif x == 'BrkFace':
        r = 2
    elif x == 'BrkCmn':
        r = 1
    else:
        r = 0
    return r
_input1['MasVnrType_int'] = _input1['MasVnrType'].apply(MasVnrTypeToInt)

def BsmtFinType1ToInt(x):
    if x == 'GLQ':
        r = 6
    elif x == 'ALQ':
        r = 5
    elif x == 'BLQ':
        r = 4
    elif x == 'Rec':
        r = 3
    elif x == 'LwQ':
        r = 2
    elif x == 'Unf':
        r = 1
    else:
        r = 0
    return r
_input1['BsmtFinType1_int'] = _input1['BsmtFinType1'].apply(BsmtFinType1ToInt)
_input1['BsmtFinType1_Unf'] = 1 * (_input1['BsmtFinType1'] == 'Unf')
_input1['HasWoodDeck'] = (_input1['WoodDeckSF'] == 0) * 1
_input1['HasOpenPorch'] = (_input1['OpenPorchSF'] == 0) * 1
_input1['HasEnclosedPorch'] = (_input1['EnclosedPorch'] == 0) * 1
_input1['Has3SsnPorch'] = (_input1['3SsnPorch'] == 0) * 1
_input1['HasScreenPorch'] = (_input1['ScreenPorch'] == 0) * 1
_input1['YearsSinceRemodel'] = _input1['YrSold'].astype(int) - _input1['YearRemodAdd'].astype(int)
_input1['Total_Home_Quality'] = _input1['OverallQual'] + _input1['OverallCond']

def addlogs(res, ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pd.Series(np.log(1.01 + res[l])).values)
        res.columns.values[m] = l + '_log'
        m += 1
    return res
loglist = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'YearRemodAdd', 'TotalSF']
_input1 = addlogs(_input1, loglist)

def getdummies(res, ls):

    def encode(encode_df):
        encode_df = np.array(encode_df)
        enc = OneHotEncoder()
        le = LabelEncoder()