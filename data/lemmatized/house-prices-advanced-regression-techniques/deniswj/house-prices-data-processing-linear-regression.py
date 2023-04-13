import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from tqdm.notebook import tqdm
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')

def housing_clean(train_data, test_data, eda=False):
    """
    Provides basic cleaning and processing of the House Prices data.
    param train_data: pandas.DataFrame, input training data
    param test_data: pandas.DataFrame, input test data
    param eda: bool, set True to return combined dataframe before scaling for EDA purposes.
    """
    train = _input1.copy()
    test = _input0.copy()
    categorical = ['MSZoning', 'Street', 'Alley', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'HeatingQC', 'CentralAir', 'Electrical', 'GarageType', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
    numerical = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
    years = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']
    ordinal = {'LotShape': ['IR3', 'IR2', 'IR1', 'Reg'], 'LandSlope': ['Sev', 'Mod', 'Gtl'], 'ExterQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'], 'ExterCond': ['Po', 'Fa', 'TA', 'Gd', 'Ex'], 'BsmtQual': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], 'BsmtCond': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], 'BsmtExposure': ['None', 'No', 'Mn', 'Av', 'Gd'], 'BsmtFinType1': ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'], 'BsmtFinType2': ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'], 'Heating': ['Floor', 'Grav', 'Wall', 'OthW', 'GasW', 'GasA'], 'HeatingQC': ['Po', 'Fa', 'TA', 'Gd', 'Ex'], 'KitchenQual': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], 'Functional': ['None', 'Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'], 'FireplaceQu': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], 'GarageFinish': ['None', 'Unf', 'RFn', 'Fin'], 'GarageQual': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], 'GarageCond': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], 'PavedDrive': ['N', 'P', 'Y'], 'PoolQC': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex']}
    to_drop = ['MSSubClass', 'TotalBsmtSF', 'Utilities']
    exterior_cols = train['Exterior1st'].unique()[~np.isin(train['Exterior1st'].unique(), ['Other', 'None'])]
    train_target = pd.DataFrame(np.log1p(train['SalePrice']))
    if not eda:
        train = train.drop('SalePrice', axis=1, inplace=False)
    dfs = {'train': train, 'test': test}
    for (df, df_obj) in dfs.items():
        categorical_copy = categorical.copy()
        numerical_copy = numerical.copy()
        years_copy = years.copy()
        to_drop_copy = to_drop.copy()
        for col in years_copy:
            dfs[df][col + 'Rel'] = 2010 - dfs[df][col]
            to_drop_copy.append(col)
            numerical_copy.append(col + 'Rel')
        for (incorr, corr) in zip(['CmentBd', 'Wd Shng', 'Brk Cmn'], ['CemntBd', 'WdShing', 'BrkComm']):
            dfs[df].loc[dfs[df]['Exterior2nd'] == incorr, 'Exterior2nd'] = corr
        dfs[df]['MoSoldRel'] = dfs[df]['YrSoldRel'] * 12 + (12 - dfs[df]['MoSold'])
        numerical_copy.remove('YrSoldRel')
        to_drop_copy.extend(['YrSoldRel', 'MoSold'])
        numerical_copy.append('MoSoldRel')
        na_replace = {**{col: 'None' for col in categorical_copy}, **{col: 0 for col in numerical_copy}, **{col: 'None' for col in ordinal.keys()}, **{col: 2010 for col in years_copy}}
        dfs[df] = dfs[df].fillna(na_replace, inplace=False)
        for (col, order) in ordinal.items():
            dfs[df][col] = pd.Categorical(dfs[df][col], categories=order)
        condition_cols = dfs[df]['Condition1'].unique()[~np.isin(dfs[df]['Condition1'].unique(), ['Norm'])]
        condition_df = pd.DataFrame(np.zeros((dfs[df].shape[0], len(condition_cols)), dtype='int'), columns=condition_cols)
        for col in condition_cols:
            cond1_tf = np.where(dfs[df]['Condition1'] == col, 1, 0)
            cond2_tf = np.where(dfs[df]['Condition2'] == col, 1, 0)
            condition_df[col] = condition_df[col] + cond1_tf + cond2_tf
        dfs[df] = pd.concat([df_obj, condition_df], axis=1)
        to_drop_copy.extend(['Condition1', 'Condition2'])
        categorical_copy.remove('Condition1')
        categorical_copy.remove('Condition2')
        exterior_df = pd.DataFrame(np.zeros((dfs[df].shape[0], len(exterior_cols)), dtype='int'), columns=exterior_cols)
        for col in exterior_cols:
            ext1_tf = np.where(dfs[df]['Exterior1st'] == col, 1, 0)
            ext2_tf = np.where(dfs[df]['Exterior2nd'] == col, 1, 0)
            exterior_df[col] = exterior_df[col] + ext1_tf + ext2_tf
        dfs[df] = pd.concat([df_obj, exterior_df], axis=1)
        to_drop_copy.extend(['Exterior1st', 'Exterior2nd'])
        categorical_copy.remove('Exterior1st')
        categorical_copy.remove('Exterior2nd')
        dfs[df] = dfs[df].drop(columns=to_drop_copy, axis=1, inplace=False)
        mapper_df = DataFrameMapper([([col], OrdinalEncoder(categories=[cat], dtype='uint8')) for (col, cat) in ordinal.items()], df_out=True)
        ord_df = mapper_df.fit_transform(dfs[df].copy())
        for col in ordinal.keys():
            dfs[df][col] = ord_df[col]
    categorical = categorical_copy
    if not eda:
        train_len = dfs['train'].shape[0]
        combine = dfs['train'].append(dfs['test'])
        combine = pd.get_dummies(combine, columns=categorical, drop_first=True)
        combine['TotalSF'] = combine['GrLivArea'] + combine['BsmtFinSF1'] + combine['BsmtFinSF2'] + combine['BsmtUnfSF'] + combine['1stFlrSF'] + combine['2ndFlrSF'] + combine['LowQualFinSF']
        to_log = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'ScreenPorch', 'TotalSF']
        for col in to_log:
            combine[col] = np.log1p(combine[col])
        ss = StandardScaler().fit_transform(combine.iloc[:, 1:])
        combine = pd.DataFrame(ss, columns=combine.columns[1:])
        (train, test) = (combine.iloc[:train_len, :], combine.iloc[train_len:, :])
    else:
        (train, test) = (dfs['train'], dfs['test'])
    test_id = dfs['test']['Id'].astype('int')
    return (train, test, train_target, test_id.values)
(train, test, train_target, test_id) = housing_clean(_input1, _input0)
train.shape
(train_cp, _, train_target_cp, _) = housing_clean(_input1, _input0)
(mean_scores, mean_stds) = ([], [])
linreg = Ridge(random_state=42, alpha=100)