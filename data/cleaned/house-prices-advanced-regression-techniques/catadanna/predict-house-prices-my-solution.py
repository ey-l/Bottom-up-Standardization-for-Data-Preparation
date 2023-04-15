import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import csv
import re
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_squared_error, mean_squared_log_error
FILE_TRAIN = '/kaggle/input/house-prices-advanced-regression-techniques/train.csv'
FILE_TEST = '/kaggle/input/house-prices-advanced-regression-techniques/test.csv'
NAN_STR_REPLACEMENT = '8888.0'
NAN_INT_REPLACEMENT = 8888
NAN_FLOAT_REPLACEMENT = 8888.0
STRING_COLUMNS = ['MSZoning', 'Alley', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'Fence', 'MiscFeature', 'YrSold', 'SaleType', 'SaleCondition']
NUMERIC_COLUMNS = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'nb_missing_values']
PARAMS_HUBER = dict()
PARAMS_HUBER['epsilon'] = 1.35
PARAMS_HUBER['max_iter'] = 50000
PARAMS_HUBER['alpha'] = 0.0001

def readData(inFile, sep=','):
    df_op = pd.read_csv(filepath_or_buffer=inFile, low_memory=False, encoding='utf-8', sep=sep)
    return df_op

def set_missing_values_column(df):
    df['nb_missing_values'] = df.shape[1] - df.count(axis=1)
    return df

def vectorize_exp(x):
    r = math.exp(x)
    return r

def vectorize_expm1(x):
    r = math.expm1(x)
    return r

def vectorize_log(x):
    r = math.log(x)
    return r

def vectorize_log1p(x):
    r = math.log1p(x)
    return r

def vectorize_int(x):
    return int(x)

def convert_data_to_numeric(df, enc_big=None):
    MSZoning_mapping = {'C (all)': 0, 'FV': 1, 'RH': 2, 'RL': 3, 'RM': 4}
    Street_mapping = {'Grvl': 0, 'Pave': 1}
    Alley_mapping = {'Grvl': 0, 'Pave': 1}
    LotShape_mapping = {'IR1': 0, 'IR2': 1, 'IR3': 2, 'Reg': 3}
    LandContour_mapping = {'Bnk': 0, 'HLS': 1, 'Low': 2, 'Lvl': 3}
    Utilities_mapping = {'AllPub': 0, 'NoSeWa': 1}
    LotConfig_mapping = {'Corner': 0, 'CulDSac': 1, 'FR2': 2, 'FR3': 3, 'Inside': 4}
    LandSlope_mapping = {'Gtl': 0, 'Mod': 1, 'Sev': 2}
    Condition1_mapping = {'Artery': 0, 'Feedr': 1, 'Norm': 2, 'PosA': 3, 'PosN': 4, 'RRAe': 5, 'RRAn': 6, 'RRNe': 7, 'RRNn': 8}
    Condition2_mapping = {'Artery': 0, 'Feedr': 1, 'Norm': 2, 'PosA': 3, 'PosN': 4, 'RRAe': 5, 'RRAn': 6, 'RRNn': 7}
    BldgType_mapping = {'1Fam': 0, '2fmCon': 1, 'Duplex': 2, 'Twnhs': 3, 'TwnhsE': 4}
    HouseStyle_mapping = {'1.5Fin': 0, '1.5Unf': 1, '1Story': 2, '2.5Fin': 3, '2.5Unf': 4, '2Story': 5, 'SFoyer': 6, 'SLvl': 7}
    RoofStyle_mapping = {'Flat': 0, 'Gable': 1, 'Gambrel': 2, 'Hip': 3, 'Mansard': 4, 'Shed': 5}
    RoofMatl_mapping = {'ClyTile': 0, 'CompShg': 1, 'Membran': 2, 'Metal': 3, 'Roll': 4, 'Tar&Grv': 5, 'WdShake': 6, 'WdShngl': 7}
    MasVnrType_mapping = {'BkrCmn': 0, 'BrkFace': 1, 'None': 2, 'Stone': 3}
    ExterQual_mapping = {'Ex': 0, 'Fa': 1, 'Gd': 2, 'TA': 3}
    ExterCond_mapping = {'Ex': 0, 'Fa': 1, 'Gd': 2, 'Po': 3, 'TA': 4}
    Foundation_mapping = {'BrkTill': 0, 'CBlock': 1, 'PConc': 2, 'Slab': 3, 'Stone': 4, 'Wood': 5}
    BsmtQual_mapping = {'Ex': 0, 'Fa': 1, 'Gd': 2, 'TA': 3}
    BsmtCond_mapping = {'Fa': 1, 'Gd': 2, 'Po': 3, 'TA': 4}
    BsmtExposure_mapping = {'Av': 0, 'Gd': 1, 'Mn': 2, 'No': 3}
    BsmtFinType1_mapping = {'ALQ': 0, 'BLQ': 1, 'GLQ': 2, 'LwQ': 3, 'Rec': 4, 'Unf': 5}
    BsmtFinType2_mapping = {'ALQ': 0, 'BLQ': 1, 'GLQ': 2, 'LwQ': 3, 'Rec': 4, 'Unf': 5}
    Heating_mapping = {'Floor': 0, 'GasA': 1, 'GasW': 2, 'Grav': 3, 'OthW': 4, 'Wall': 5}
    HeatingQC_mapping = {'Ex': 0, 'Fa': 1, 'Gd': 2, 'Po': 3, 'TA': 4}
    CentralAir_mapping = {'N': 0, 'Y': 1}
    Electrical_mapping = {'FuseA': 0, 'FuseF': 1, 'FuseP': 2, 'Mix': 3, 'SBrkr': 4}
    KitchenQual_mapping = {'Ex': 0, 'Fa': 1, 'Gd': 2, 'TA': 3}
    Functional_mapping = {'Maj1': 0, 'Maj2': 1, 'Min1': 2, 'Min2': 3, 'Mod': 4, 'Sev': 5, 'Typ': 6}
    FireplaceQu_mapping = {'Ex': 0, 'Fa': 1, 'Gd': 2, 'Po': 3, 'TA': 4}
    GarageType_mapping = {'2Types': 0, 'Attchd': 1, 'Basement': 2, 'BuiltIn': 3, 'CatPort': 4, 'Detchd': 5}
    GarageFinish_mapping = {'Fin': 0, 'RFn': 1, 'Unf': 2}
    GarageQual_mapping = {'Ex': 0, 'Fa': 1, 'Gd': 2, 'Po': 3, 'TA': 4}
    GarageCond_mapping = {'Ex': 0, 'Fa': 1, 'Gd': 2, 'Po': 3, 'TA': 4}
    PavedDrive_mapping = {'N': 0, 'P': 1, 'Y': 2}
    PoolQC_mapping = {'Ex': 0, 'Fa': 1, 'Gd': 2}
    Fence_mapping = {'GdPrv': 0, 'GdWo': 1, 'MnPrv': 2, 'MnWw': 3}
    MiscFeature_mapping = {'Gar2': 0, 'Othr': 1, 'Shed': 2, 'TenC': 3}
    SaleType_mapping = {'COD': 0, 'Con': 1, 'ConLD': 2, 'ConLI': 3, 'ConLw': 4, 'CWD': 5, 'New': 6, 'Oth': 7, 'WD': 8}
    SaleCondition_mapping = {'Abnorml': 0, 'AdjLand': 1, 'Alloca': 2, 'Family': 3, 'Normal': 4, 'Partial': 5}
    enc_big_new = enc_big
    if enc_big == None:
        Neighborhood_mapping = dict(zip(df.Neighborhood.dropna().unique(), range(len(df.Neighborhood.dropna().unique()))))
        Exterior1st_mapping = dict(zip(df.Exterior1st.dropna().unique(), range(len(df.Exterior1st.dropna().unique()))))
        Exterior2nd_mapping = dict(zip(df.Exterior2nd.dropna().unique(), range(len(df.Exterior2nd.dropna().unique()))))
        enc_big_new = {'Neighborhood': Neighborhood_mapping, 'Exterior1st': Exterior1st_mapping, 'Exterior2nd': Exterior2nd_mapping}
    else:
        Neighborhood_mapping = enc_big_new['Neighborhood']
        Exterior1st_mapping = enc_big_new['Exterior1st']
        Exterior2nd_mapping = enc_big_new['Exterior2nd']
    df['MSZoning'] = df.loc[df.MSZoning.notnull(), 'MSZoning'].map(MSZoning_mapping)
    df['Street'] = df.loc[df.Street.notnull(), 'Street'].map(Street_mapping)
    df['Alley'] = df.loc[df.Alley.notnull(), 'Alley'].map(Alley_mapping)
    df['LotShape'] = df.loc[df.LotShape.notnull(), 'LotShape'].map(LotShape_mapping)
    df['LandContour'] = df.loc[df.LandContour.notnull(), 'LandContour'].map(LandContour_mapping)
    df['Utilities'] = df.loc[df.Utilities.notnull(), 'Utilities'].map(Utilities_mapping)
    df['LotConfig'] = df.loc[df.LotConfig.notnull(), 'LotConfig'].map(LotConfig_mapping)
    df['LandSlope'] = df.loc[df.LandSlope.notnull(), 'LandSlope'].map(LandSlope_mapping)
    df['Neighborhood'] = df.loc[df.Neighborhood.notnull(), 'Neighborhood'].map(Neighborhood_mapping)
    df['Condition1'] = df.loc[df.Condition1.notnull(), 'Condition1'].map(Condition1_mapping)
    df['Condition2'] = df.loc[df.Condition2.notnull(), 'Condition2'].map(Condition2_mapping)
    df['BldgType'] = df.loc[df.BldgType.notnull(), 'BldgType'].map(BldgType_mapping)
    df['HouseStyle'] = df.loc[df.HouseStyle.notnull(), 'HouseStyle'].map(HouseStyle_mapping)
    df['RoofStyle'] = df.loc[df.RoofStyle.notnull(), 'RoofStyle'].map(RoofStyle_mapping)
    df['RoofMatl'] = df.loc[df.RoofMatl.notnull(), 'RoofMatl'].map(RoofMatl_mapping)
    df['Exterior1st'] = df.loc[df.Exterior1st.notnull(), 'Exterior1st'].map(Exterior1st_mapping)
    df['Exterior2nd'] = df.loc[df.Exterior2nd.notnull(), 'Exterior2nd'].map(Exterior2nd_mapping)
    df['MasVnrType'] = df.loc[df.MasVnrType.notnull(), 'MasVnrType'].map(MasVnrType_mapping)
    df['ExterQual'] = df.loc[df.ExterQual.notnull(), 'ExterQual'].map(ExterQual_mapping)
    df['ExterCond'] = df.loc[df.ExterCond.notnull(), 'ExterCond'].map(ExterCond_mapping)
    df['Foundation'] = df.loc[df.Foundation.notnull(), 'Foundation'].map(Foundation_mapping)
    df['BsmtQual'] = df.loc[df.BsmtQual.notnull(), 'BsmtQual'].map(BsmtQual_mapping)
    df['BsmtCond'] = df.loc[df.BsmtCond.notnull(), 'BsmtCond'].map(BsmtCond_mapping)
    df['BsmtExposure'] = df.loc[df.BsmtExposure.notnull(), 'BsmtExposure'].map(BsmtExposure_mapping)
    df['BsmtFinType1'] = df.loc[df.BsmtFinType1.notnull(), 'BsmtFinType1'].map(BsmtFinType1_mapping)
    df['BsmtFinType2'] = df.loc[df.BsmtFinType2.notnull(), 'BsmtFinType2'].map(BsmtFinType2_mapping)
    df['Heating'] = df.loc[df.Heating.notnull(), 'Heating'].map(Heating_mapping)
    df['HeatingQC'] = df.loc[df.HeatingQC.notnull(), 'HeatingQC'].map(HeatingQC_mapping)
    df['CentralAir'] = df.loc[df.CentralAir.notnull(), 'CentralAir'].map(CentralAir_mapping)
    df['Electrical'] = df.loc[df.Electrical.notnull(), 'Electrical'].map(Electrical_mapping)
    df['KitchenQual'] = df.loc[df.KitchenQual.notnull(), 'KitchenQual'].map(KitchenQual_mapping)
    df['Functional'] = df.loc[df.Functional.notnull(), 'Functional'].map(Functional_mapping)
    df['FireplaceQu'] = df.loc[df.FireplaceQu.notnull(), 'FireplaceQu'].map(FireplaceQu_mapping)
    df['GarageType'] = df.loc[df.GarageType.notnull(), 'GarageType'].map(GarageType_mapping)
    df['GarageFinish'] = df.loc[df.GarageFinish.notnull(), 'GarageFinish'].map(GarageFinish_mapping)
    df['GarageQual'] = df.loc[df.GarageQual.notnull(), 'GarageQual'].map(GarageQual_mapping)
    df['GarageCond'] = df.loc[df.GarageCond.notnull(), 'GarageCond'].map(GarageCond_mapping)
    df['PavedDrive'] = df.loc[df.PavedDrive.notnull(), 'PavedDrive'].map(PavedDrive_mapping)
    df['PoolQC'] = df.loc[df.PoolQC.notnull(), 'PoolQC'].map(PoolQC_mapping)
    df['Fence'] = df.loc[df.Fence.notnull(), 'Fence'].map(Fence_mapping)
    df['MiscFeature'] = df.loc[df.MiscFeature.notnull(), 'MiscFeature'].map(MiscFeature_mapping)
    df['SaleType'] = df.loc[df.SaleType.notnull(), 'SaleType'].map(SaleType_mapping)
    df['SaleCondition'] = df.loc[df.SaleCondition.notnull(), 'SaleCondition'].map(SaleCondition_mapping)
    return (df, enc_big_new)

def get_encoded_ii(df, ii=None, Y=None):
    df.loc[:, STRING_COLUMNS] = df[STRING_COLUMNS].fillna(NAN_FLOAT_REPLACEMENT).astype('int32').astype('category')
    df.loc[:, NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].fillna(NAN_FLOAT_REPLACEMENT).astype('int32')
    cols = df.columns
    if ii is None:
        ii = IterativeImputer(missing_values=NAN_INT_REPLACEMENT, initial_strategy='most_frequent')