import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import math, re
from scipy import stats
from scipy.stats import norm, skew
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from nltk.tokenize import sent_tokenize, word_tokenize
sns.set_style('whitegrid')
from sklearn.preprocessing import LabelEncoder
pd.options.mode.chained_assignment = None
from PIL import Image
path = '../input/houses-2/houses.jfif'
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
data = pd.concat([_input1, _input0], axis=0, sort=False).reset_index(drop=True)
data = data.drop('SalePrice', axis=1, inplace=False)

def multi_plotting(df, feature):
    fig = plt.figure(constrained_layout=True, figsize=(12, 8))
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
    ax1 = fig.add_subplot(grid[0, :2])
    ax1.set_title('Histogram')
    sns.distplot(df.loc[:, feature], norm_hist=True, ax=ax1)
    ax2 = fig.add_subplot(grid[1, :2])
    ax2.set_title('QQ_plot')
    stats.probplot(df.loc[:, feature], plot=ax2)
    ax3 = fig.add_subplot(grid[:, 2])
    ax3.set_title('Box Plot')
    sns.boxplot(df.loc[:, feature], orient='v', ax=ax3)
    print('Skewness: ' + str(_input1['SalePrice'].skew().round(3)))
    print('Kurtosis: ' + str(_input1['SalePrice'].kurt().round(3)))
multi_plotting(_input1, 'SalePrice')
_input1['SalePrice'] = np.log1p(_input1['SalePrice'])
multi_plotting(_input1, 'SalePrice')

def missing_rate(frame):
    TL = frame.isna().sum() / len(frame)
    TCMR = TL[TL > 0.6].index
    frame = frame.drop(TCMR, axis=1, inplace=False)
    print(f'Colums to be dropped:{TCMR}')
missing_rate(_input1)
missing_rate(data)

def corr_heat(frame):
    correlation = frame.corr()
    (f, ax) = plt.subplots(figsize=(30, 20))
    mask = np.triu(correlation)
    sns.heatmap(correlation, annot=True, mask=mask, ax=ax, cmap='viridis')
    (bottom, top) = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)

def box_plot(column1, column2, column3, data):
    (f, (ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(15, 5))
    sns.boxplot(y='SalePrice', x=column1, data=data, ax=ax1)
    sns.boxplot(y='SalePrice', x=column2, data=data, ax=ax2)
    sns.boxplot(y='SalePrice', x=column3, data=data, ax=ax3)

def finding_zeros(frame):
    Zeros = frame[frame == 0].count().sort_values(ascending=False)
    print(Zeros, '/', end='')
numeric = data.dtypes[data.dtypes != 'object'].index
tnumeric = _input1.dtypes[_input1.dtypes != 'object'].index
integer = data[numeric]
tinteger = _input1[tnumeric]
integer = integer.fillna(0, inplace=False)
tinteger = tinteger.fillna(0, inplace=False)
corr_heat(tinteger)
FC = tinteger.corr()
Target = tinteger.corr()['SalePrice'].to_frame().reset_index()
FR = FC.unstack().to_frame(name='Correlation')
Feature = FR[(FR['Correlation'] >= 0.8) & (FR['Correlation'] < 1)].sort_values(by='Correlation', ascending=False).reset_index()
Feature.head(5)
Final = Feature.merge(Target, left_on='level_1', right_on='index')
Final
tinteger = tinteger.drop(['GarageArea', 'GarageYrBlt', 'TotRmsAbvGrd', 'Id'], axis=1, inplace=False)
integer = integer.drop(['GarageArea', 'GarageYrBlt', 'TotRmsAbvGrd', 'Id'], axis=1, inplace=False)
t = tinteger[tinteger == 0].count().sort_values(ascending=False).head(25)
d = integer[integer == 0].count().sort_values(ascending=False).head(25)
source = {'Train': t, 'Data': d}
Zeros = pd.DataFrame.from_dict(source)
Zeros.sort_values(by=['Train', 'Data'], ascending=False)
tinteger['Porch'] = tinteger['3SsnPorch'] + tinteger['ScreenPorch'] + tinteger['EnclosedPorch'] + tinteger['OpenPorchSF']
tinteger['YNPorch'] = tinteger['Porch'].apply(lambda x: 1 if x > 0 else 0)
tinteger = tinteger.drop(['3SsnPorch', 'ScreenPorch', 'EnclosedPorch', 'OpenPorchSF', 'PoolArea'], axis=1, inplace=False)
integer['Porch'] = integer['3SsnPorch'] + integer['ScreenPorch'] + integer['EnclosedPorch'] + integer['OpenPorchSF']
integer['YNPorch'] = integer['Porch'].apply(lambda x: 1 if x > 0 else 0)
integer = integer.drop(['3SsnPorch', 'ScreenPorch', 'EnclosedPorch', 'OpenPorchSF', 'PoolArea'], axis=1, inplace=False)
tinteger
tinteger[['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']].tail(5)

def cat_basmentfin(df):
    df['Bsmntbuilt'] = 0
    A = df[(df['BsmtFinSF1'] > 0) | (df['BsmtFinSF2'] > 0) & (df['BsmtUnfSF'] >= 0)].index
    B = df[(df['BsmtFinSF1'] == 0) & (df['BsmtFinSF2'] == 0) & (df['BsmtUnfSF'] > 0)].index
    C = df[(df['BsmtFinSF1'] == 0) & (df['BsmtFinSF2'] == 0) & (df['BsmtUnfSF'] == 0)].index
    df.loc[A, 'Bsmntbuilt'] = 2
    df.loc[B, 'Bsmntbuilt'] = 1
    df.loc[C, 'Bsmntbuilt'] = 0
    df = df.drop(['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF'], axis=1, inplace=False)
tinteger.iloc[:, 10:]
cat_basmentfin(tinteger)
cat_basmentfin(integer)

def basmentbath(df):
    No_BsmtFullBath = df[(df['BsmtFullBath'] == 0) & (df['BsmtHalfBath'] == 0)].index
    Only_halfBath = df[(df['BsmtFullBath'] == 0) & (df['BsmtHalfBath'] == 1)].index
    Two_halfBath = df[(df['BsmtFullBath'] == 0) & (df['BsmtHalfBath'] == 2)].index
    Only_FullBath = df[(df['BsmtFullBath'] == 1) & (df['BsmtHalfBath'] == 0)].index
    One_Full_Half = df[(df['BsmtFullBath'] == 1) & (df['BsmtHalfBath'] == 1)].index
    Two_Full = df[(df['BsmtFullBath'] == 2) & (df['BsmtHalfBath'] == 0)].index
    Three_Full = df[(df['BsmtFullBath'] == 3) & (df['BsmtHalfBath'] == 0)].index
    df['BsmtBathCat'] = 0
    df.loc[No_BsmtFullBath, 'BsmtBathCat'] = 0
    df.loc[Only_halfBath, 'BsmtBathCat'] = 1
    df.loc[Two_halfBath, 'BsmtBathCat'] = 2
    df.loc[Only_FullBath, 'BsmtBathCat'] = 3
    df.loc[One_Full_Half, 'BsmtBathCat'] = 4
    df.loc[Two_Full, 'BsmtBathCat'] = 5
    df.loc[Three_Full, 'BsmtBathCat'] = 6
    df = df.drop(['BsmtFullBath', 'BsmtHalfBath'], axis=1, inplace=False)
basmentbath(tinteger)
basmentbath(integer)
tinteger.iloc[:, 13:]
integer['2ndFloor'] = integer['2ndFlrSF'].apply(lambda x: 0 if x == 0 else 1)
tinteger['2ndFloor'] = tinteger['2ndFlrSF'].apply(lambda x: 0 if x == 0 else 1)
integer = integer.drop(['1stFlrSF', '2ndFlrSF'], axis=1, inplace=False)
tinteger = tinteger.drop(['1stFlrSF', '2ndFlrSF'], axis=1, inplace=False)
tinteger = tinteger.join(_input1['Neighborhood'])
lot_mean = tinteger.groupby('Neighborhood')['LotFrontage'].mean().round().to_dict()
tinteger.loc[(tinteger['LotFrontage'] == 0) & tinteger['Neighborhood'].isin(lot_mean.keys()), 'LotFrontage'] = tinteger['Neighborhood'].map(lot_mean)
tinteger = tinteger.drop('Neighborhood', axis=1, inplace=False)
integer = integer.join(data['Neighborhood'])
lot_mean = integer.groupby('Neighborhood')['LotFrontage'].mean().round().to_dict()
integer.loc[(integer['LotFrontage'] == 0) & integer['Neighborhood'].isin(lot_mean.keys()), 'LotFrontage'] = integer['Neighborhood'].map(lot_mean)
integer = integer.drop('Neighborhood', axis=1, inplace=False)
tinteger.iloc[:, 13:]
integer = integer.drop(['LowQualFinSF', 'MiscVal'], axis=1, inplace=False)
tinteger = tinteger.drop(['LowQualFinSF', 'MiscVal'], axis=1, inplace=False)
t = tinteger[tinteger == 0].count().sort_values(ascending=False).head(14)
d = integer[integer == 0].count().sort_values(ascending=False).head(14)
source = {'Train': t, 'Data': d}
Zeros = pd.DataFrame.from_dict(source)
Zeros.sort_values(by=['Train', 'Data'], ascending=False)
corr_heat(tinteger)
box_plot('MoSold', 'YrSold', 'KitchenAbvGr', tinteger)
box_plot('OverallCond', 'MSSubClass', 'KitchenAbvGr', tinteger)
integer = integer.drop(['MoSold', 'YrSold', 'KitchenAbvGr'], axis=1, inplace=False)
tinteger = tinteger.drop(['MoSold', 'YrSold', 'KitchenAbvGr'], axis=1, inplace=False)
categorical = data[['KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageFinish', 'GarageCond', 'BsmtCond', 'BsmtExposure', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1', 'ExterCond', 'ExterQual', 'HeatingQC', 'LandSlope', 'LotShape', 'PavedDrive', 'Street', 'BldgType', 'Functional', 'SaleType', 'GarageType', 'Electrical', 'Foundation', 'LandContour', 'RoofMatl', 'RoofStyle']]
tcategorical = _input1[['SalePrice', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageFinish', 'GarageCond', 'BsmtCond', 'BsmtExposure', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1', 'ExterCond', 'ExterQual', 'HeatingQC', 'LandSlope', 'LotShape', 'PavedDrive', 'Street', 'BldgType', 'Functional', 'SaleType', 'GarageType', 'Electrical', 'Foundation', 'LandContour', 'RoofMatl', 'RoofStyle']]
kit = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
fire = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
garagequa = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
garage_finish = {'Fin': 3, 'RFn': 2, 'Unf': 1, 'None': 0}
garagecond = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
bsmtcond = {'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
bsmtexposure = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'None': 0}
bsmtqual = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'None': 0}
bsmtype2 = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'None': 0}
bsmtype1 = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'None': 0}
extercond = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
exterqual = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
heatingqc = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
landslope = {'Gtl': 3, 'Mod': 2, 'Sev': 1}
lotshape = {'Reg': 4, 'IR1': 3, 'IR2': 2, 'IR3': 1}
pavedrive = {'Y': 3, 'P': 2, 'N': 1}
street = {'Grvl': 2, 'Pave': 1}
bldgtype = {'1Fam': 1, '2fmCon': 2, 'Duplex': 3, 'TwnhsE': 4, 'Twnhs': 5}
functional = {'Typ': 8, 'Min1': 7, 'Min2': 6, 'Mod': 5, 'Maj1': 4, 'Maj2': 3, 'Sev': 2, 'Sal': 1, 'None': 1}
saletype = {'WD': 10, 'CWD': 9, 'VWD': 8, 'New': 7, 'COD': 6, 'Con': 5, 'ConLw': 4, 'ConLI': 3, 'ConLD': 2, 'Oth': 1, 'None': 0}
garage = {'2Types': 6, 'Attchd': 5, 'Basment': 4, 'BuiltIn': 3, 'CarPort': 2, 'Detchd': 1, 'None': 0}
electrical = {'SBrkr': 5, 'FuseA': 4, 'FuseF': 3, 'FuseP': 2, 'Mix': 1, 'None': 0}
foundation = {'BrkTil': 6, 'CBlock': 5, 'PConc': 4, 'Slab': 3, 'Stone': 2, 'Wood': 1}
landcontour = {'Lvl': 4, 'Bnk': 3, 'HLS': 2, 'Low': 1}
roofmatl = {'ClyTile': 8, 'CompShg': 7, 'Membran': 6, 'Metal': 5, 'Roll': 4, 'Tar&Grv': 3, 'WdShake': 2, 'WdShngl': 1}
roofstyle = {'Flat': 6, 'Gable': 5, 'Gambrel': 4, 'Hip': 3, 'Mansard': 2, 'Shed': 1}
for i in tcategorical:
    tcategorical[i] = tcategorical[i].fillna('None', inplace=False)
for i in categorical:
    categorical[i] = categorical[i].fillna('None', inplace=False)
for i in tcategorical.columns[1:].to_list():
    v = tcategorical[i].unique()
    print(i, v)
list_dict = [kit, fire, garagequa, garage_finish, garagecond, bsmtcond, bsmtexposure, bsmtqual, bsmtype2, bsmtype1, extercond, exterqual, heatingqc, landslope, lotshape, pavedrive, street, bldgtype, functional, saletype, garage, electrical, foundation, landcontour, roofmatl, roofstyle]

def replace_t(i, diccionaty):
    tcategorical[i] = tcategorical[i].replace(diccionaty, inplace=False)

def replace(i, diccionaty):
    categorical[i] = categorical[i].replace(diccionaty, inplace=False)
for (i, j) in zip(tcategorical.columns[1:].to_list(), list_dict):
    replace_t(i, j)
for (i, j) in zip(categorical.columns.to_list(), list_dict):
    replace(i, j)
for i in categorical:
    v = categorical[i].unique()
    print(i, v)
corr_heat(tcategorical)
for i in tcategorical:
    v = tcategorical[i].value_counts().to_frame().reset_index().iloc[0].to_list()
    w = tcategorical[i].value_counts().reset_index().iloc[0][0]
    if w == 0:
        print(i, v)
tcategorical = tcategorical.drop('FireplaceQu', axis=1, inplace=False)
categorical = categorical.drop('FireplaceQu', axis=1, inplace=False)
nominal = data[['Utilities', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MSZoning', 'CentralAir', 'Condition1', 'Condition2', 'Heating', 'HouseStyle', 'LotConfig', 'Neighborhood', 'SaleCondition']]
nominal.isnull().sum().head(6)
for i in nominal:
    encoding = pd.get_dummies(nominal[i], prefix=i, drop_first=True)
    nominal = pd.concat([nominal, encoding], axis=1)
    nominal = nominal.drop(i, axis=1, inplace=False)
nominal
all_features = nominal.keys()
nominal = nominal.drop(nominal.loc[:, (nominal == 0).sum() >= nominal.shape[0] * 0.9994], axis=1)
nominal = nominal.drop(nominal.loc[:, (nominal == 1).sum() >= nominal.shape[0] * 0.9994], axis=1)
remain_features = nominal.keys()
remov_features = [st for st in all_features if st not in remain_features]
print(len(remov_features), 'features were removed:', remov_features)
print(integer.shape, categorical.shape, nominal.shape)
new_data = pd.concat([integer, categorical, nominal], axis=1, sort=False)
new_data
train_data = new_data.iloc[:1460].reset_index(drop=True)
test_data = new_data.iloc[1460:].reset_index(drop=True)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
X = train_data
y = _input1['SalePrice']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.33, random_state=42)