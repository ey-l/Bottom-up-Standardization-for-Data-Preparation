import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
combine = [_input1, _input0]
_input1.name = 'Train'
_input0.name = 'Test'
_input1.sample(10)
_input1.columns
for df in combine:
    if df.name == 'Train':
        mis_train = []
        cols = df.columns
        for col in cols:
            mis = df[col].isnull().sum()
            if mis > 0:
                print('{}: {} missing, {}%'.format(col, mis, round(mis / df.shape[0] * 100, 3)))
                mis_train.append(col)
        print('_' * 40)
        print('_' * 40)
    if df.name == 'Test':
        mis_test = []
        cols = df.columns
        for col in cols:
            mis = df[col].isnull().sum()
            if mis > 0:
                print('{}: {} missing, {}%'.format(col, mis, round(mis / df.shape[0] * 100, 3)))
                mis_test.append(col)
print('\n')
print(mis_train)
print('_' * 40)
print(mis_test)

def traintest_hist(feat, nbins):
    (fig, axes) = plt.subplots(1, 2)
    _input1[feat].hist(bins=nbins, ax=axes[0])
    _input0[feat].hist(bins=nbins, ax=axes[1])
    print('{}: {} missing, {}%'.format('Train', _input1[feat].isnull().sum(), round(_input1[feat].isnull().sum() / _input1.shape[0] * 100, 3)))
    print('{}: {} missing, {}%'.format('Test', _input0[feat].isnull().sum(), round(_input0[feat].isnull().sum() / _input0.shape[0] * 100, 3)))

def categ_dist_mis(cats):
    for cat in cats:
        print(cat)
        print('_' * 40)
        print('In train: ')
        print(_input1[cat].value_counts(dropna=False))
        print('_' * 40)
        print('In test: ')
        print(_input0[cat].value_counts(dropna=False))
        print('_' * 40)
        print('_' * 40)

def checkclean(feats):
    for feat in feats:
        print(feat)
        for df in combine:
            print('In {}'.format(df.name))
            print(df[feat].value_counts(dropna=False))
            print('_' * 40)
    print('_' * 40)
traintest_hist('LotFrontage', 50)
traintest_hist('LotArea', 50)
cats = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2']
categ_dist_mis(cats)
for df in combine:
    df.loc[(df.MSZoning == 'FV') | (df.MSZoning == 'RH') | (df.MSZoning == 'C (all)'), 'MSZoning'] = 3
    df.loc[df.MSZoning == 'RL', 'MSZoning'] = 1
    df.loc[df.MSZoning == 'RM', 'MSZoning'] = 2
    df.loc[df.Condition1 != 'Norm', 'Condition1'] = 0
    df.loc[df.Condition1 == 'Norm', 'Condition1'] = 1
    df.loc[df.LotFrontage.isnull(), 'LotFrontage'] = 0
    df.loc[df.Alley.notnull(), 'Alley'] = 1
    df.loc[df.Alley.isnull(), 'Alley'] = 0
    df.loc[df.LotShape != 'Reg', 'LotShape'] = 0
    df.loc[df.LotShape == 'Reg', 'LotShape'] = 1
    df.loc[df.LotConfig == 'Inside', 'LotConfig'] = 1
    df.loc[df.LotConfig == 'Corner', 'LotConfig'] = 2
    df.loc[df.LotConfig == 'CulDSac', 'LotConfig'] = 3
    df.loc[(df.LotConfig == 'FR2') | (df.LotConfig == 'FR3'), 'LotConfig'] = 4
    df.loc[df.LandSlope != 'Gtl', 'LandSlope'] = 0
    df.loc[df.LandSlope == 'Gtl', 'LandSlope'] = 1
    df.loc[df.LandContour != 'Lvl', 'LandContour'] = 0
    df.loc[df.LandContour == 'Lvl', 'LandContour'] = 1
print('LotFrontage')
print('In train: ')
print(_input1.LotFrontage.isnull().sum())
print('_' * 40)
print('In test: ')
print(_input0.LotFrontage.isnull().sum())
print('_' * 40)
print('_' * 40)
feats = ['MSZoning', 'Condition1', 'Alley', 'LotShape', 'LotShape', 'LotConfig', 'LandSlope', 'LandContour']
checkclean(feats)
traintest_hist('OverallQual', 10)
traintest_hist('OverallCond', 10)
traintest_hist('YearBuilt', 20)
traintest_hist('YearRemodAdd', 20)
cats = ['MSSubClass', 'BldgType', 'HouseStyle']
categ_dist_mis(cats)
for df in combine:
    df.loc[df.BldgType == '1Fam', 'BldgType'] = 1
    df.loc[(df.BldgType == '2fmCon') | (df.BldgType == 'Duplex'), 'BldgType'] = 2
    df.loc[df.BldgType == 'CulDSac', 'BldgType'] = 3
    df.loc[(df.BldgType == 'TwnhsE') | (df.BldgType == 'Twnhs'), 'BldgType'] = 4
    df.loc[df.HouseStyle == '1Story', 'HouseStyle'] = 1
    df.loc[(df.HouseStyle == 'SFoyer') | (df.HouseStyle == 'SLvl'), 'HouseStyle'] = 2
    sto = ['1.5Fin', '1.5Unf', '2.5Fin', '2.5Unf', '2Story']
    df.loc[df.HouseStyle.isin(sto), 'HouseStyle'] = 3
feats = ['BldgType', 'HouseStyle']
checkclean(feats)
traintest_hist('MasVnrArea', 20)
cats = ['Foundation', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond']
categ_dist_mis(cats)
for df in combine:
    df.loc[df.MasVnrArea == 0, 'MasVnr'] = 0
    df.loc[df.MasVnrArea > 0, 'MasVnr'] = 1
    df.loc[df.MasVnrType == 'None', 'MasVnrType'] = 0
    df.loc[(df.MasVnrType == 'BrkFace') | (df.MasVnrType == 'BrkCmn'), 'MasVnrType'] = 1
    df.loc[df.MasVnrType == 'Stone', 'MasVnrType'] = 2
    df.loc[(df.ExterQual == 'TA') | (df.ExterQual == 'Fa'), 'ExterQual'] = 0
    df.loc[(df.ExterQual == 'Gd') | (df.ExterQual == 'Ex'), 'ExterQual'] = 1
    df.loc[(df.ExterCond == 'TA') | (df.ExterCond == 'Fa') | (df.ExterCond == 'Po'), 'ExterCond'] = 0
    df.loc[(df.ExterCond == 'Gd') | (df.ExterCond == 'Ex'), 'ExterCond'] = 1
    feat = 'Exterior1st'
    df.loc[df[feat] == 'VinylSd', feat] = 0
    df.loc[(df[feat] == 'Stucco') | (df[feat] == 'ImStucc'), feat] = 1
    df.loc[(df[feat] == 'Wd Sdng') | (df[feat] == 'WdShing') | (df[feat] == 'Plywood'), feat] = 2
    df.loc[df[feat] == 'MetalSd', feat] = 3
    df.loc[df[feat] == 'HdBoard', feat] = 4
    df.loc[(df[feat] == 'BrkFace') | (df[feat] == 'BrkComm'), feat] = 5
    df.loc[(df[feat] == 'CemntBd') | (df[feat] == 'AsbShng') | (df[feat] == 'AsphShn') | (df[feat] == 'CBlock') | (df[feat] == 'Stone'), feat] = 1
    feat = 'Exterior2nd'
    df.loc[df[feat] == 'VinylSd', feat] = 0
    df.loc[(df[feat] == 'Stucco') | (df[feat] == 'ImStucc'), feat] = 1
    df.loc[(df[feat] == 'Wd Sdng') | (df[feat] == 'Wd Shng') | (df[feat] == 'Plywood'), feat] = 2
    df.loc[df[feat] == 'MetalSd', feat] = 3
    df.loc[df[feat] == 'HdBoard', feat] = 4
    df.loc[(df[feat] == 'BrkFace') | (df[feat] == 'Brk Cmn'), feat] = 1
    df.loc[(df[feat] == 'CmentBd') | (df[feat] == 'AsbShng') | (df[feat] == 'AsphShn') | (df[feat] == 'CBlock') | (df[feat] == 'Stone') | (df[feat] == 'Other'), feat] = 1
feats = ['MasVnr', 'MasVnrType', 'ExterQual', 'ExterCond', 'Exterior1st', 'Exterior2nd']
checkclean(feats)
traintest_hist('BsmtFinSF1', 30)
traintest_hist('BsmtFinSF2', 30)
traintest_hist('BsmtUnfSF', 30)
traintest_hist('TotalBsmtSF', 30)
traintest_hist('BsmtFullBath', 5)
traintest_hist('BsmtHalfBath', 5)
cats = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
categ_dist_mis(cats)
for df in combine:
    df['BsmFinSFTot'] = df['BsmtFinSF1'] + df['BsmtFinSF2']
    df['BsmBath'] = df['BsmtFullBath'] + df['BsmtHalfBath']
    df.loc[df.BsmBath == 0, 'BsmBath'] = 0
    df.loc[df.BsmBath > 0, 'BsmBath'] = 1
    fil = df.BsmtQual.isnull() & df.BsmtCond.isnull() & df.BsmtExposure.isnull() & df.BsmtFinType1.isnull() & df.BsmtFinType2.isnull()
    df['MisBsm'] = 0
    df.loc[fil, 'MisBsm'] = 1
    df.loc[fil, 'BsmtQual'] = -99
    df.loc[(df.BsmtQual == 'Fa') | (df.BsmtQual == 'TA'), 'BsmtQual'] = 1
    df.loc[df.BsmtQual == 'Gd', 'BsmtQual'] = 2
    df.loc[df.BsmtQual == 'Ex', 'BsmtQual'] = 3
    df.loc[fil, 'BsmtCond'] = -99
    df.loc[(df.BsmtCond == 'Fa') | (df.BsmtCond == 'TA') | (df.BsmtCond == 'Po'), 'BsmtCond'] = 1
    df.loc[df.BsmtCond == 'Gd', 'BsmtCond'] = 2
    df.loc[df.BsmtCond == 'Ex', 'BsmtCond'] = 3
    df.loc[fil, 'BsmtExposure'] = -99
    df.loc[df.BsmtExposure == 'Gd', 'BsmtExposure'] = 1
    df.loc[df.BsmtExposure == 'Av', 'BsmtExposure'] = 2
    df.loc[df.BsmtExposure == 'Mn', 'BsmtExposure'] = 3
    df.loc[df.BsmtExposure == 'No', 'BsmtExposure'] = 4
    df.loc[fil, 'BsmtFinType1'] = -99
    df.loc[df.BsmtFinType1 == 'Unf', 'BsmtFinType1'] = 1
    df.loc[df.BsmtFinType1 == 'GLQ', 'BsmtFinType1'] = 2
    df.loc[df.BsmtFinType1 == 'ALQ', 'BsmtFinType1'] = 3
    df.loc[df.BsmtFinType1 == 'BLQ', 'BsmtFinType1'] = 4
    df.loc[df.BsmtFinType1 == 'Rec', 'BsmtFinType1'] = 5
    df.loc[df.BsmtFinType1 == 'LwQ', 'BsmtFinType1'] = 6
    df.loc[fil, 'BsmtFinType2'] = -99
    df.loc[(df.BsmtFinType2 != 'Unf') & (df.BsmtFinType2 != -99), 'BsmtFinType2'] = 0
    df.loc[df.BsmtFinType2 == 'Unf', 'BsmtFinType2'] = 1
feats = ['BsmBath', 'MisBsm', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
checkclean(feats)
traintest_hist('BsmFinSFTot', 30)
_input1[_input1.BsmFinSFTot == 0]['MisBsm'].value_counts(dropna=False)
fil = _input1.BsmFinSFTot + _input1.BsmtUnfSF != _input1.TotalBsmtSF
df[fil].shape
cats = ['Utilities', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical']
categ_dist_mis(cats)
for df in combine:
    df.loc[df.HeatingQC == 'Ex', 'HeatingQC'] = 1
    df.loc[df.HeatingQC == 'Gd', 'HeatingQC'] = 2
    df.loc[df.HeatingQC == 'TA', 'HeatingQC'] = 3
    df.loc[(df.HeatingQC == 'Fa') | (df.HeatingQC == 'Po'), 'HeatingQC'] = 4
    df.loc[df.CentralAir == 'N', 'CentralAir'] = 0
    df.loc[df.CentralAir == 'Y', 'CentralAir'] = 1
    fil = (df.Electrical == 'FuseA') | (df.Electrical == 'FuseF') | (df.Electrical == 'FuseP') | (df.Electrical == 'Mix')
    df.loc[fil, 'Electrical'] = 0
    df.loc[df.Electrical == 'SBrkr', 'Electrical'] = 1
feats = ['HeatingQC', 'CentralAir', 'Electrical']
checkclean(feats)
traintest_hist('1stFlrSF', 30)
traintest_hist('2ndFlrSF', 30)
traintest_hist('LowQualFinSF', 30)
traintest_hist('GrLivArea', 30)
traintest_hist('FullBath', 10)
traintest_hist('HalfBath', 10)
traintest_hist('BedroomAbvGr', 10)
traintest_hist('KitchenAbvGr', 10)
traintest_hist('TotRmsAbvGrd', 20)
cats = ['KitchenQual', 'Functional']
categ_dist_mis(cats)
for df in combine:
    df['2ndFlr'] = 0
    df.loc[df['2ndFlrSF'] > 0, '2ndFlr'] = 1
    df['BathsTot'] = df.FullBath + df.HalfBath
    df.loc[df.KitchenQual == 'Ex', 'KitchenQual'] = 1
    df.loc[df.KitchenQual == 'Gd', 'KitchenQual'] = 2
    df.loc[(df.KitchenQual == 'TA') | (df.KitchenQual == 'Fa'), 'KitchenQual'] = 3
    fun = ['Min2', 'Min1', 'Mod', 'Maj1', 'Sev', 'Maj2']
    fil = df.Functional.isin(fun)
    df.loc[fil, 'Functional'] = 0
    df.loc[df.Functional == 'Typ', 'Functional'] = 1
feats = ['2ndFlr', 'BathsTot', 'KitchenQual', 'Functional']
checkclean(feats)
fil = _input1['1stFlrSF'] + _input1['2ndFlrSF'] != _input1.GrLivArea
df[fil].shape
fil = _input1['BedroomAbvGr'] + _input1['KitchenAbvGr'] != _input1.TotRmsAbvGrd
df[fil].shape
traintest_hist('Fireplaces', 10)
traintest_hist('GarageYrBlt', 30)
traintest_hist('GarageCars', 10)
traintest_hist('GarageArea', 30)
cats = ['FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive']
categ_dist_mis(cats)
for df in combine:
    df.loc[df['Fireplaces'] > 0, 'Fireplaces'] = 1
    df.loc[df['Fireplaces'] == 0, 'Fireplaces'] = 0
    df.loc[(df.Fireplaces == 0) & df.FireplaceQu.isnull(), 'FireplaceQu'] = -99
    df.loc[(df.FireplaceQu == 'Ex') | (df.FireplaceQu == 'Gd'), 'FireplaceQu'] = 1
    df.loc[df.FireplaceQu == 'TA', 'FireplaceQu'] = 2
    df.loc[(df.FireplaceQu == 'Fa') | (df.FireplaceQu == 'Po'), 'FireplaceQu'] = 3
    fil = df.GarageYrBlt.isnull() & df.GarageType.isnull() & df.GarageFinish.isnull() & df.GarageQual.isnull() & df.GarageCond.isnull()
    df['MisGarage'] = 0
    df.loc[fil, 'MisGarage'] = 1
    df.loc[df.GarageYrBlt > 2200, 'GarageYrBlt'] = 2007
    df.loc[fil, 'GarageYrBlt'] = -99
    gty = ['Attchd', 'BuiltIn', 'Basment', '2Types']
    fil1 = df.GarageType.isin(gty)
    fil2 = (df.GarageType == 'Detchd') | (df.GarageType == 'CarPort')
    df.loc[fil, 'GarageType'] = -99
    df.loc[fil1, 'GarageType'] = 0
    df.loc[fil2, 'GarageType'] = 1
    df.loc[fil, 'GarageFinish'] = -99
    df.loc[df.GarageFinish == 'Unf', 'GarageFinish'] = 1
    df.loc[df.GarageFinish == 'RFn', 'GarageFinish'] = 2
    df.loc[df.GarageFinish == 'Fin', 'GarageFinish'] = 3
    df.loc[df.PavedDrive == 'Y', 'PavedDrive'] = 1
    df.loc[df.PavedDrive == 'P', 'PavedDrive'] = 1
    df.loc[df.PavedDrive == 'N', 'PavedDrive'] = 0
feats = ['Fireplaces', 'FireplaceQu', 'MisGarage', 'GarageType', 'GarageFinish', 'PavedDrive']
checkclean(feats)
traintest_hist('GarageYrBlt', 30)
traintest_hist('WoodDeckSF', 30)
traintest_hist('OpenPorchSF', 30)
traintest_hist('EnclosedPorch', 30)
traintest_hist('3SsnPorch', 30)
traintest_hist('ScreenPorch', 30)
traintest_hist('PoolArea', 30)
cats = ['PoolQC', 'Fence']
categ_dist_mis(cats)
for df in combine:
    df['Porch'] = df.ScreenPorch + df['3SsnPorch'] + df.EnclosedPorch + df.OpenPorchSF + df.WoodDeckSF
    df.loc[df.Fence.isnull(), 'Fence'] = 0
    df.loc[(df.Fence == 'MnPrv') | (df.Fence == 'GdPrv'), 'Fence'] = 1
    df.loc[(df.Fence == 'GdWo') | (df.Fence == 'MnWw'), 'Fence'] = 2
feats = ['Fence']
checkclean(feats)
traintest_hist('Porch', 30)
traintest_hist('MiscVal', 30)
traintest_hist('YrSold', 10)
traintest_hist('MoSold', 12)
cats = ['MiscFeature', 'SaleType', 'SaleCondition']
categ_dist_mis(cats)
for df in combine:
    war = ['WD', 'CWD', 'VWD']
    con = ['Con', 'ConLw', 'ConLI', 'ConLD', 'COD']
    oth = ['New', 'Oth']
    df.loc[df.SaleType.isin(war), 'SaleType'] = 1
    df.loc[df.SaleType.isin(con), 'SaleType'] = 2
    df.loc[df.SaleType.isin(oth), 'SaleType'] = 3
    nnor = ['Abnorml', 'AdjLand', 'Alloca', 'Family', 'Partial']
    df.loc[df.SaleCondition.isin(nnor), 'SaleCondition'] = 0
    df.loc[df.SaleCondition == 'Normal', 'SaleCondition'] = 1
feats = ['SaleType', 'SaleCondition']
checkclean(feats)
irrs = ['Street', 'Condition2', 'Utilities', 'RoofMatl', 'RoofStyle', 'Heating', 'LowQualFinSF', 'GarageQual', 'GarageCond', 'PoolArea', 'PoolQC', 'MiscVal', 'MiscFeature', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'ScreenPorch', 'EnclosedPorch', '3SsnPorch', 'OpenPorchSF', 'WoodDeckSF']
print('Before: {} features in train and {} in test'.format(len(_input1.columns), len(_input0.columns)))
for df in combine:
    for irr in irrs:
        del df[irr]
print('After: {} features in train and {} in test'.format(len(_input1.columns), len(_input0.columns)))
for df in combine:
    if df.name == 'Train':
        mis_train = []
        cols = df.columns
        for col in cols:
            mis = df[col].isnull().sum()
            if mis > 0:
                print('{}: {} missing, {}%'.format(col, mis, round(mis / df.shape[0] * 100, 3)))
                mis_train.append(col)
        print('_' * 40)
        print('_' * 40)
    if df.name == 'Test':
        mis_test = []
        cols = df.columns
        for col in cols:
            mis = df[col].isnull().sum()
            if mis > 0:
                print('{}: {} missing, {}%'.format(col, mis, round(mis / df.shape[0] * 100, 3)))
                mis_test.append(col)
print('\n')
print(mis_train)
print('_' * 40)
print(mis_test)

def find_segment(df, feat):
    mis = df[feat].isnull().sum()
    cols = df.columns
    seg = []
    for col in cols:
        vc = df[df[feat].isnull()][col].value_counts(dropna=False).iloc[0]
        if vc == mis:
            seg.append(col)
    return seg

def find_mode(df, feat):
    md = df[df[feat].isnull()][find_segment(df, feat)].dropna(axis=1).mode()
    md = pd.merge(df, md, how='inner')[feat].mode().iloc[0]
    return md

def find_median(df, feat):
    md = df[df[feat].isnull()][find_segment(df, feat)].dropna(axis=1).mode()
    md = pd.merge(df, md, how='inner')[feat].median()
    return md

def similar_mode(df, col, feats):
    sm = df[df[col].isnull()][feats]
    md = pd.merge(df, sm, how='inner')[col].mode().iloc[0]
    return md

def similar_median(df, col, feats):
    sm = df[df[col].isnull()][feats]
    md = pd.merge(df, sm, how='inner')[col].median()
    return md
md = find_mode(_input1, 'MasVnrType')
print('MasVnrType {}'.format(md))
_input1[['MasVnrType']] = _input1[['MasVnrType']].fillna(md)
md = find_mode(_input1, 'MasVnr')
print('MasVnr {}'.format(md))
_input1[['MasVnr']] = _input1[['MasVnr']].fillna(md)
simi = ['BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2']
md = similar_mode(_input1, 'BsmtExposure', simi)
print('BsmtExposure {}'.format(md))
_input1[['BsmtExposure']] = _input1[['BsmtExposure']].fillna(md)
simi = ['HeatingQC', 'CentralAir']
md = similar_mode(_input1, 'Electrical', simi)
print('Electrical {}'.format(md))
_input1[['Electrical']] = _input1[['Electrical']].fillna(md)
cols = _input1.columns
print('Start printing the missing values...')
for col in cols:
    mis = _input1[col].isnull().sum()
    if mis > 0:
        print('{}: {} missing, {}%'.format(col, mis, round(mis / _input1.shape[0] * 100, 3)))
print('...done printing the missing values')
md = find_mode(_input0, 'MSZoning')
print('MSZoning {}'.format(md))
_input0[['MSZoning']] = _input0[['MSZoning']].fillna(md)
simi = ['ExterQual', 'ExterCond']
md = similar_mode(_input0, 'Exterior1st', simi)
print('Exterior1st {}'.format(md))
_input0[['Exterior1st']] = _input0[['Exterior1st']].fillna(md)
simi = ['ExterQual', 'ExterCond']
md = similar_mode(_input0, 'Exterior2nd', simi)
print('Exterior2nd {}'.format(md))
_input0[['Exterior2nd']] = _input0[['Exterior2nd']].fillna(md)
md = find_mode(_input0, 'MasVnrType')
print('MasVnrType {}'.format(md))
_input0[['MasVnrType']] = _input0[['MasVnrType']].fillna(md)
md = find_mode(_input0, 'MasVnr')
print('MasVnr {}'.format(md))
_input0[['MasVnr']] = _input0[['MasVnr']].fillna(md)
simi = ['BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
md = similar_mode(_input0, 'BsmtQual', simi)
print('BsmtQual {}'.format(md))
_input0[['BsmtQual']] = _input0[['BsmtQual']].fillna(md)
simi = ['BsmtQual', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
md = similar_mode(_input0, 'BsmtCond', simi)
print('BsmtCond {}'.format(md))
_input0[['BsmtCond']] = _input0[['BsmtCond']].fillna(md)
simi = ['BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2']
md = similar_mode(_input0, 'BsmtExposure', simi)
print('BsmtExposure {}'.format(md))
_input0[['BsmtExposure']] = _input0[['BsmtExposure']].fillna(md)
_input0[['BsmtUnfSF', 'TotalBsmtSF']] = _input0[['BsmtUnfSF', 'TotalBsmtSF']].fillna(-99)
md = _input0.KitchenQual.mode().iloc[0]
print('KitchenQual {}'.format(md))
_input0[['KitchenQual']] = _input0[['KitchenQual']].fillna(md)
_input0[['Functional']] = _input0[['Functional']].fillna(1)
simi = ['YearBuilt']
md = similar_median(_input0, 'GarageYrBlt', simi)
print('GarageYrBlt {}'.format(md))
_input0[['GarageYrBlt']] = _input0[['GarageYrBlt']].fillna(md)
simi = ['YearBuilt', 'GarageYrBlt']
md = similar_median(_input0, 'GarageFinish', simi)
print('GarageFinish {}'.format(md))
_input0[['GarageFinish']] = _input0[['GarageFinish']].fillna(md)
simi = ['GarageType', 'MisGarage']
md = similar_mode(_input0, 'GarageCars', simi)
print('GarageCars {}'.format(md))
_input0[['GarageCars']] = _input0[['GarageCars']].fillna(md)
simi = ['GarageType', 'MisGarage', 'GarageCars']
md = similar_median(_input0, 'GarageArea', simi)
print('GarageArea {}'.format(md))
_input0[['GarageArea']] = _input0[['GarageArea']].fillna(md)
simi = ['SaleCondition']
md = similar_mode(_input0, 'SaleType', simi)
print('SaleType {}'.format(md))
_input0[['SaleType']] = _input0[['SaleType']].fillna(md)
_input0[['BsmFinSFTot', 'BsmBath']] = _input0[['BsmFinSFTot', 'BsmBath']].fillna(-99)
cols = _input0.columns
print('Start printing the missing values...')
for col in cols:
    mis = _input0[col].isnull().sum()
    if mis > 0:
        print('{}: {} missing, {}%'.format(col, mis, round(mis / _input0.shape[0] * 100, 3)))
print('...done printing the missing values')
strings = ['MSSubClass']
ints = ['Alley', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope', 'Condition1', 'BldgType', 'HouseStyle', 'Exterior1st', 'Exterior2nd', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'HeatingQC', 'CentralAir', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'PavedDrive', 'Fence', 'SaleType', 'SaleCondition']
for df in combine:
    df[strings] = df[strings].astype(str)
    df[ints] = df[ints].astype(int)