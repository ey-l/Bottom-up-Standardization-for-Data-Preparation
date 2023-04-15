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
mydata = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
mydata.drop(mydata[(mydata['OverallQual'] < 5) & (mydata['SalePrice'] > 200000)].index, inplace=True)
mydata.drop(mydata[(mydata['GrLivArea'] > 4000) & (mydata['SalePrice'] < 300000)].index, inplace=True)
mydata.reset_index(drop=True, inplace=True)
mydata['MSSubClass'] = mydata['MSSubClass'].apply(str)
mydata['YrSold'] = mydata['YrSold'].astype(str)
mydata['MoSold'] = mydata['MoSold'].astype(str)

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
mydata = fill_missings(mydata)
mydata['TotalSF'] = mydata['TotalBsmtSF'] + mydata['1stFlrSF'] + mydata['2ndFlrSF']

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
mydata['ExterQual'] = mydata['ExterQual'].apply(QualToInt)
mydata['ExterCond'] = mydata['ExterCond'].apply(QualToInt)
mydata['KitchenQual'] = mydata['KitchenQual'].apply(QualToInt)
mydata['HeatingQC'] = mydata['HeatingQC'].apply(QualToInt)
mydata['BsmtQual'] = mydata['BsmtQual'].apply(QualToInt)
mydata['BsmtCond'] = mydata['BsmtCond'].apply(QualToInt)
mydata['FireplaceQu'] = mydata['FireplaceQu'].apply(QualToInt)
mydata['GarageQual'] = mydata['GarageQual'].apply(QualToInt)
mydata['PoolQC'] = mydata['PoolQC'].apply(QualToInt)

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
mydata['LandSlope'] = mydata['LandSlope'].apply(SlopeToInt)
mydata['CentralAir'] = mydata['CentralAir'].apply(lambda x: 0 if x == 'N' else 1)
mydata['Street'] = mydata['Street'].apply(lambda x: 0 if x == 'Pave' else 1)
mydata['PavedDrive'] = mydata['PavedDrive'].apply(lambda x: 0 if x == 'Y' else 1)

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
mydata['GarageFinish'] = mydata['GarageFinish'].apply(GFinishToInt)

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
mydata['BsmtExposure'] = mydata['BsmtExposure'].apply(BsmtExposureToInt)

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
mydata['Functional_int'] = mydata['Functional'].apply(FunctionalToInt)

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
mydata['HouseStyle_int'] = mydata['HouseStyle'].apply(HouseStyleToInt)
mydata['HouseStyle_1st'] = 1 * (mydata['HouseStyle'] == '1Story')
mydata['HouseStyle_2st'] = 1 * (mydata['HouseStyle'] == '2Story')
mydata['HouseStyle_15st'] = 1 * (mydata['HouseStyle'] == '1.5Fin')

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
mydata['Foundation_int'] = mydata['Foundation'].apply(FoundationToInt)

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
mydata['MasVnrType_int'] = mydata['MasVnrType'].apply(MasVnrTypeToInt)

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
mydata['BsmtFinType1_int'] = mydata['BsmtFinType1'].apply(BsmtFinType1ToInt)
mydata['BsmtFinType1_Unf'] = 1 * (mydata['BsmtFinType1'] == 'Unf')
mydata['HasWoodDeck'] = (mydata['WoodDeckSF'] == 0) * 1
mydata['HasOpenPorch'] = (mydata['OpenPorchSF'] == 0) * 1
mydata['HasEnclosedPorch'] = (mydata['EnclosedPorch'] == 0) * 1
mydata['Has3SsnPorch'] = (mydata['3SsnPorch'] == 0) * 1
mydata['HasScreenPorch'] = (mydata['ScreenPorch'] == 0) * 1
mydata['YearsSinceRemodel'] = mydata['YrSold'].astype(int) - mydata['YearRemodAdd'].astype(int)
mydata['Total_Home_Quality'] = mydata['OverallQual'] + mydata['OverallCond']

def addlogs(res, ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pd.Series(np.log(1.01 + res[l])).values)
        res.columns.values[m] = l + '_log'
        m += 1
    return res
loglist = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'YearRemodAdd', 'TotalSF']
mydata = addlogs(mydata, loglist)

def getdummies(res, ls):

    def encode(encode_df):
        encode_df = np.array(encode_df)
        enc = OneHotEncoder()
        le = LabelEncoder()