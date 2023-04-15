import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy import stats
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
train.head()
train.info()
X = train.drop('SalePrice', axis=1)
y = train['SalePrice']
numerical_cols = [col for col in X.columns if X[col].dtype in ['float64', 'int64']]
ordinal_cols = ['MSZoning', 'LotShape', 'Utilities', 'LandSlope', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence']
nominal_cols = ['Street', 'Alley', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'HouseStyle', 'BldgType', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'CentralAir', 'Electrical', 'GarageType', 'MiscFeature', 'SaleType', 'SaleCondition']
train_num = train[numerical_cols].copy()
train_ord = train[ordinal_cols].copy()
train_nom = train[nominal_cols].copy()

def find_nan_cols(df):
    null = df.isnull().sum()
    missing_df = pd.concat([null], axis=1, keys=['nancount'])
    return missing_df[missing_df.nancount > 0]
find_nan_cols(train_num)
train_num.drop(['LotFrontage', 'GarageYrBlt'], axis=1, inplace=True)
numerical_cols = train_num.columns

def Fill_num(df):
    meds = {}
    for col in find_nan_cols(df).index:
        df.fillna(df[col].median(), inplace=True)
        meds[col] = df[col].median()
    return meds
meds = Fill_num(train_num)
from sklearn.feature_selection import mutual_info_regression as MIR
mi_score = MIR(np.array(train_num), y.ravel())
mi_score
mi_score_selected_index = np.where(mi_score > 0.2)[0]
mi_score_selected_index
train_num_ = train_num.iloc[:, mi_score_selected_index]
corrmat = train_num_.corr()
(f, ax) = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)
numerical_final_cols = ['OverallQual', 'YearBuilt', 'TotalBsmtSF', 'GrLivArea', 'FullBath', 'GarageCars']
train_num_array = np.array(train_num[numerical_final_cols])

def Fill_ord(df):
    mods = {}
    some_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence']
    for col in find_nan_cols(df).index:
        if col in some_cols:
            df[col] = df[col].replace(np.nan, 'NOTFOUND')
        else:
            df[col] = df[col].replace(np.nan, df[col].mode()[0])
            mods[col] = df[col].mode()[0]
    return mods
mods = Fill_ord(train_ord)
from sklearn.preprocessing import OrdinalEncoder

def OrdinalEncoding(col, cat):
    ord_enc = OrdinalEncoder(categories=[cat])
    return ord_enc.fit_transform(col)

def Ordinal_Encoding(df):
    categories = {'MSZoning': ['A', 'C (all)', 'FV', 'I', 'RH', 'RL', 'RP', 'RM'], 'LotShape': ['IR3', 'IR2', 'IR1', 'Reg'], 'Utilities': ['ELO', 'NoSeWa', 'NoSewr', 'AllPub'], 'LandSlope': ['Sev', 'Mod', 'Gtl'], 'ExterQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'], 'ExterCond': ['Po', 'Fa', 'TA', 'Gd', 'Ex'], 'BsmtQual': ['NOTFOUND', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], 'BsmtCond': ['NOTFOUND', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], 'BsmtExposure': ['NOTFOUND', 'No', 'Mn', 'Av', 'Gd'], 'BsmtFinType1': ['NOTFOUND', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'], 'BsmtFinType2': ['NOTFOUND', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'], 'HeatingQC': ['Po', 'Fa', 'TA', 'Gd', 'Ex'], 'KitchenQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'], 'Functional': ['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'], 'FireplaceQu': ['NOTFOUND', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], 'GarageFinish': ['NOTFOUND', 'Unf', 'RFn', 'Fin'], 'GarageQual': ['NOTFOUND', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], 'GarageCond': ['NOTFOUND', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], 'PavedDrive': ['N', 'P', 'Y'], 'PoolQC': ['NOTFOUND', 'Fa', 'TA', 'Gd', 'Ex'], 'Fence': ['NOTFOUND', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv']}
    for col in df.columns:
        df[col] = OrdinalEncoding(df[[col]], categories[col])
Ordinal_Encoding(train_ord)
oridnal_cols_final = ['Utilities', 'ExterQual', 'BsmtQual', 'KitchenQual', 'FireplaceQu', 'GarageFinish', 'PoolQC']
train_ord_array = np.array(train_ord[oridnal_cols_final])

def Fill_nom(df):
    for col in find_nan_cols(df).index:
        df[col] = df[col].replace(np.nan, df[col].mode()[0])
Fill_nom(train_nom)
nominal_cols_final = ['Street', 'Neighborhood', 'Condition1', 'HouseStyle', 'BldgType', 'RoofStyle', 'Exterior1st', 'MasVnrType', 'Heating', 'CentralAir', 'Electrical', 'SaleType', 'SaleCondition']
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
oh_enc = OneHotEncoder(sparse=False)
column_transform = make_column_transformer((oh_enc, nominal_cols_final))