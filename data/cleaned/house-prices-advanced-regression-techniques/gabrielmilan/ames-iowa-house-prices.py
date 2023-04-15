import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
import scipy.stats as st
from scipy import stats
from scipy.special import boxcox1p
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
df_train.columns
df_train['SalePrice'].describe()
df_train.describe()
df_train.head()
y = df_train['SalePrice']
plt.figure(1)
plt.title('Johnson SU')
sns.distplot(y, kde=False, fit=st.johnsonsu)
plt.figure(2)
plt.title('Normal')
sns.distplot(y, kde=False, fit=st.norm)
plt.figure(3)
plt.title('Log Normal')
sns.distplot(y, kde=False, fit=st.lognorm)

df_train = df_train.drop(df_train[df_train.SalePrice >= 700000].index)
df_train = df_train.drop(df_train[df_train.LotFrontage >= 200].index)
df_train = df_train.drop(df_train[df_train.LotArea >= 100000].index)
df_train = df_train.drop(df_train[(df_train.OverallQual == 4) & (df_train.SalePrice > 200000)].index)
df_train = df_train.drop(df_train[(df_train.OverallQual == 8) & (df_train.SalePrice > 500000)].index)
df_train = df_train.drop(df_train[(df_train.OverallQual == 10) & (df_train.SalePrice < 300000)].index)
df_train = df_train.drop(df_train[(df_train.OverallCond == 2) & (df_train.SalePrice > 300000)].index)
df_train = df_train.drop(df_train[(df_train.OverallCond == 5) & (df_train.SalePrice > 700000)].index)
df_train = df_train.drop(df_train[(df_train.OverallCond == 6) & (df_train.SalePrice > 700000)].index)
df_train = df_train.drop(df_train[(df_train.YearBuilt <= 1900) & (df_train.SalePrice > 200000)].index)
df_train = df_train.drop(df_train[df_train.MasVnrArea >= 1200].index)
df_train = df_train.drop(df_train[df_train.BsmtFinSF1 >= 3000].index)
df_train = df_train.drop(df_train[df_train.BsmtFinSF2 >= 1200].index)
df_train = df_train.drop(df_train[df_train.TotalBsmtSF >= 4000].index)
df_train = df_train.drop(df_train[df_train['1stFlrSF'] >= 4000].index)
df_train = df_train.drop(df_train[(df_train.LowQualFinSF > 500) & (df_train.SalePrice > 400000)].index)
df_train = df_train.drop(df_train[df_train.GrLivArea >= 4000].index)
df_train = df_train.drop(df_train[df_train.BedroomAbvGr >= 7].index)
df_train = df_train.drop(df_train[df_train.KitchenAbvGr < 1].index)
df_train = df_train.drop(df_train[df_train.TotRmsAbvGrd > 12].index)
df_train = df_train.drop(df_train[df_train.Fireplaces > 2].index)
df_train = df_train.drop(df_train[df_train.GarageCars > 3].index)
df_train = df_train.drop(df_train[(df_train.GarageArea > 1200) & (df_train.SalePrice < 300000)].index)
df_train = df_train.drop(df_train[df_train.WoodDeckSF > 700].index)
df_train = df_train.drop(df_train[df_train.OpenPorchSF > 450].index)
df_train = df_train.drop(df_train[df_train.EnclosedPorch > 450].index)
df_train = df_train.drop(df_train[df_train['3SsnPorch'] > 350].index)
full = pd.concat([df_train, df_test], ignore_index=True)
full.drop(['SalePrice'], axis=1, inplace=True)
words_set = set()
for data in full['MiscFeature']:
    if not pd.isnull(data):
        for word in data.split(' '):
            words_set.add(word)
for feature in words_set:
    count = 0
    for data in full['MiscFeature']:
        if not pd.isnull(data):
            if feature in data:
                count += 1
    print('Category {}: {} samples'.format(feature, count))

def add_columns(dataframe):
    add_categorical = ['Othr', 'TenC', 'Gar2', 'Shed']
    for column in add_categorical:
        data = dataframe['MiscVal'][dataframe['MiscFeature'] == column]
        dataframe[column] = pd.Series(data, index=dataframe.index).fillna(value=0)
    dataframe.drop(columns=['MiscFeature'], inplace=True)
    dataframe.drop(columns=['MiscVal'], inplace=True)
add_columns(full)
print('Full data shape is {}'.format(full.shape))
plt.scatter(range(0, len(df_train['SalePrice'])), df_train['SalePrice'])

categorical_features = ['Alley', 'BldgType', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'CentralAir', 'Condition1', 'Condition2', 'Electrical', 'ExterCond', 'Exterior1st', 'Exterior2nd', 'ExterQual', 'Fence', 'FireplaceQu', 'Foundation', 'Functional', 'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType', 'Heating', 'HeatingQC', 'HouseStyle', 'KitchenQual', 'LandContour', 'LandSlope', 'LotConfig', 'LotShape', 'MasVnrType', 'MSZoning', 'Neighborhood', 'PavedDrive', 'PoolQC', 'RoofMatl', 'RoofStyle', 'SaleCondition', 'SaleType', 'Street', 'Utilities']
numerical_features = ['1stFlrSF', '2ndFlrSF', '3SsnPorch', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF', 'EnclosedPorch', 'Fireplaces', 'FullBath', 'GarageArea', 'GarageCars', 'GarageYrBlt', 'GrLivArea', 'HalfBath', 'LotArea', 'LotFrontage', 'LowQualFinSF', 'MasVnrArea', 'MoSold', 'MSSubClass', 'OpenPorchSF', 'OverallCond', 'OverallQual', 'PoolArea', 'ScreenPorch', 'TotalBsmtSF', 'TotRmsAbvGrd', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd', 'YrSold', 'BedroomAbvGr', 'KitchenAbvGr', 'Othr', 'TenC', 'Gar2', 'Shed']

def createAgeFeatures(dataframe):
    dataframe['GarageAge'] = dataframe['GarageYrBlt'].apply(lambda x: 2019 - x)
    dataframe['RemodelAge'] = dataframe['YearRemodAdd'].apply(lambda x: 2019 - x)
    dataframe['HouseAge'] = dataframe['YearBuilt'].apply(lambda x: 2019 - x)

def createPoolFeature(dataframe):
    dataframe['Pool'] = dataframe['PoolArea'].apply(lambda x: x != 0).map({True: 1, False: 0})

def handleNumericalFeatures(dataframe):
    createAgeFeatures(dataframe)
    createPoolFeature(dataframe)
handleNumericalFeatures(full)
print('Full data shape is {}'.format(full.shape))

def mapSimpleMapping(dataframe):
    dataframe['ExterCond'] = dataframe['ExterCond'].map({'Ex': 2, 'Gd': 1, 'TA': 0, 'Fa': -1, 'Po': -2})
    dataframe['ExterQual'] = dataframe['ExterQual'].map({'Ex': 2, 'Gd': 1, 'TA': 0, 'Fa': -1, 'Po': -2})
    dataframe['HeatingQC'] = dataframe['HeatingQC'].map({'Ex': 2, 'Gd': 1, 'TA': 0, 'Fa': -1, 'Po': -2})
    dataframe['KitchenQual'] = dataframe['KitchenQual'].map({'Ex': 2, 'Gd': 1, 'TA': 0, 'Fa': -1, 'Po': -2})
    dataframe['PavedDrive'] = dataframe['PavedDrive'].map({'Y': 1, 'P': 0.5, 'N': 0})
    dataframe['Street'] = dataframe['Street'].map({'Grvl': -1, 'Pave': 1})

def mapSimpleWithNan(dataframe):
    dataframe['BsmtCond'] = dataframe['BsmtCond'].map({'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0, 'NA': -1}).fillna(-1)
    dataframe['BsmtExposure'] = dataframe['BsmtExposure'].map({'Gd': 3, 'Av': 2, 'Mn': 1, 'No': 0, 'NA': -1}).fillna(-1)
    dataframe['BsmtQual'] = dataframe['BsmtQual'].map({'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0, 'NA': -1}).fillna(-1)
    dataframe['FireplaceQu'] = dataframe['FireplaceQu'].map({'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0, 'NA': -1}).fillna(-1)
    dataframe['GarageCond'] = dataframe['GarageCond'].map({'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0, 'NA': -1}).fillna(-1)
    dataframe['GarageFinish'] = dataframe['GarageFinish'].map({'Fin': 2, 'RFn': 1, 'Unf': 0, 'NA': -1}).fillna(-1)
    dataframe['GarageQual'] = dataframe['GarageQual'].map({'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0, 'NA': -1}).fillna(-1)
    dataframe['PoolQC'] = dataframe['PoolQC'].map({'Ex': 3, 'Gd': 2, 'TA': 1, 'Fa': 0, 'NA': -1}).fillna(-1)

def mapBoolean(dataframe):
    dataframe['CentralAir'] = dataframe['CentralAir'].map({'N': 0, 'Y': 1})

def handleMappedCategorical(dataframe):
    mapSimpleMapping(dataframe)
    mapSimpleWithNan(dataframe)
    mapBoolean(dataframe)
handleMappedCategorical(full)
check_plot_features = ['Alley', 'BldgType', 'BsmtFinType1', 'BsmtFinType2', 'Condition1', 'Condition2', 'Electrical', 'Exterior1st', 'Exterior2nd', 'Fence', 'Foundation', 'Functional', 'GarageType', 'Heating', 'HouseStyle', 'LandContour', 'LandSlope', 'LotConfig', 'LotShape', 'MasVnrType', 'MSZoning', 'RoofMatl', 'RoofStyle', 'SaleCondition', 'SaleType']
one_hots = ['BldgType', 'BsmtFinType1', 'BsmtFinType2', 'Condition1', 'Condition2', 'Exterior1st', 'Exterior2nd', 'Foundation', 'GarageType', 'Heating', 'HouseStyle', 'LandContour', 'LandSlope', 'LotConfig', 'MasVnrType', 'MSZoning', 'RoofMatl', 'RoofStyle', 'SaleCondition', 'SaleType']

def doOneHot(dataframe):
    try:
        for category in one_hots:
            df = pd.Categorical(dataframe[category])
            dfDummies = pd.get_dummies(df, prefix=category)
            dataframe = pd.concat([dataframe, dfDummies], axis=1)
            if category not in ['Condition1', 'Condition2']:
                dataframe.drop(columns=[category], inplace=True)
        return dataframe
    except KeyError:
        print('Oops! Category {} not found! Probably this has already been done...'.format(category))
        return dataframe
full = doOneHot(full)
print('Full data shape is {}'.format(full.shape))

def handleAlley(dataframe):
    dataframe['Alley'] = dataframe['Alley'].map({'Grvl': 0, 'Pave': 1})

def handleConditions(dataframe):
    handler = {'Normal': 0, 'RRNn': 1, 'RRNe': 1, 'PosN': 2, 'Artery': 3, 'Feedr': 3, 'RRAn': 3, 'PosA': 3, 'RRAe': 3}
    dataframe['Condition1'] = dataframe['Condition1'].map(handler)
    dataframe['Condition2'] = dataframe['Condition2'].map(handler)

def handleElectrical(dataframe):
    handler = {'Mix': 0, 'FuseP': 1, 'FuseF': 2, 'FuseA': 3, 'SBrkr': 4}
    dataframe['Electrical'] = dataframe['Electrical'].map(handler)

def handleFunctional(dataframe):
    handler = {'Typ': 0, 'Min1': -1, 'Min2': -2, 'Mod': -3, 'Maj1': -4, 'Maj2': -5, 'Sev': -6, 'Sal': -7}
    dataframe['Functional'] = dataframe['Functional'].map(handler)

def handleLotShape(dataframe):
    handler = {'Reg': 0, 'IR1': -1, 'IR2': -2, 'IR3': -3}
    dataframe['LotShape'] = dataframe['LotShape'].map(handler)

def handlePostOneHots(dataframe):
    handleAlley(dataframe)
    handleConditions(dataframe)
    handleElectrical(dataframe)
    handleFunctional(dataframe)
    handleLotShape(dataframe)
handlePostOneHots(full)
print('Full data shape is {}'.format(full.shape))

def handleFence(dataframe):
    dataframe['FencePrivacy'] = dataframe['Fence'].map({'MnPrv': 0, 'GdPrv': 1}).fillna(-1)
    dataframe['FenceWood'] = dataframe['Fence'].map({'MnWw': 0, 'GdWo': 1}).fillna(-1)
    dataframe.drop(columns=['Fence'], inplace=True)
    return dataframe
full = handleFence(full)
print('Full data shape is {}'.format(full.shape))

def handleUtilities(dataframe):
    dataframe['Electricity'] = dataframe['Utilities'].map({'AllPub': 1, 'NoSewr': 1, 'NoSeWa': 1, 'ELO': 1}).fillna(0)
    dataframe['Gas'] = dataframe['Utilities'].map({'AllPub': 1, 'NoSewr': 1, 'NoSeWa': 1, 'ELO': 0}).fillna(0)
    dataframe['Water'] = dataframe['Utilities'].map({'AllPub': 1, 'NoSewr': 1, 'NoSeWa': 0, 'ELO': 0}).fillna(0)
    dataframe['Septic Tank'] = dataframe['Utilities'].map({'AllPub': 1, 'NoSewr': 0, 'NoSeWa': 0, 'ELO': 0}).fillna(0)
    dataframe.drop(columns=['Utilities'], inplace=True)
    return dataframe
full = handleUtilities(full)
print('Full data shape is {}'.format(full.shape))
geo_heatmap = {'Neighborhood': ['Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr', 'Crawfor', 'Edwards', 'Gilbert', 'IDOTRR', 'MeadowV', 'Mitchel', 'NAmes', 'NoRidge', 'NPkVill', 'NridgHt', 'NWAmes', 'OldTown', 'SWISU', 'Sawyer', 'SawyerW', 'Somerst', 'StoneBr', 'Timber', 'Veenker'], 'Latitude': [42.0563761, 42.0218678, 42.052795, 42.024546, 42.0360959, 42.0214232, 42.028025, 42.0154024, 42.1068177, 42.0204395, 41.997282, 41.9903084, 42.046618, 42.048164, 42.0258352, 42.0597732, 42.0457802, 42.029046, 42.0266187, 42.0295218, 42.034611, 42.0508817, 42.0595539, 41.9999732, 42.0413042], 'Longitude': [-93.6466598, -93.6702853, -93.6310097, -93.6545201, -93.6575849, -93.6584089, -93.6093286, -93.6875441, -93.6553512, -93.6243787, -93.6138098, -93.603242, -93.6362807, -93.6496766, -93.6613958, -93.65166, -93.6472075, -93.6165288, -93.6486541, -93.7102833, -93.7024257, -93.6485768, -93.6365891, -93.6518812, -93.6524905]}
train_neighborhood = set(df_train['Neighborhood'].tolist())
geo_dataframe = pd.DataFrame.from_dict(geo_heatmap)
geo_dataframe = geo_dataframe[geo_dataframe['Neighborhood'].isin(train_neighborhood)]
geo_dataframe['SalePrice'] = pd.Series(df_train.groupby(['Neighborhood']).mean()['SalePrice'].values, index=geo_dataframe.index)
import folium
from folium.plugins import HeatMap
max_amount = float(geo_dataframe['SalePrice'].max())
hmap = folium.Map(location=[42.045042, -93.6473567], zoom_start=12)
hm_wide = HeatMap(list(zip(geo_dataframe.Latitude.values, geo_dataframe.Longitude.values, geo_dataframe.SalePrice.values)), min_opacity=0.4, max_val=max_amount, radius=17, blur=15, max_zoom=1)
hmap.add_child(hm_wide)
lat_dict = {}
lon_dict = {}
for i in range(len(geo_heatmap['Neighborhood'])):
    neighborhood = geo_heatmap['Neighborhood'][i]
    lat = geo_heatmap['Latitude'][i]
    lon = geo_heatmap['Longitude'][i]
    lat_dict[neighborhood] = lat
    lon_dict[neighborhood] = lon

def add_lat_lon_columns(df):
    df['Latitude'] = df['Neighborhood'].map(lat_dict)
    df['Longitude'] = df['Neighborhood'].map(lon_dict)
    df.drop(columns=['Neighborhood'], inplace=True)
add_lat_lon_columns(full)
print('Full data shape is {}'.format(full.shape))
missing_values = full.isnull().sum()
missing_values[missing_values > 0].sort_values(ascending=False)
full['KitchenQual'] = full['KitchenQual'].fillna(0)
missing_values = full.isnull().sum()
missing_values[missing_values > 0].sort_values(ascending=False)
full[full['GarageArea'].isnull()]
new = full.iloc[2576].copy()
new['GarageArea'] = 0
new['GarageCars'] = 0
new['GarageCond'] = -1
new['GarageFinish'] = -1
new['GarageQual'] = -1
full.iloc[2576] = new.copy()
missing_values = full.isnull().sum()
missing_values[missing_values > 0].sort_values(ascending=False)
full['Electrical'] = full['Electrical'].fillna(2)
missing_values = full.isnull().sum()
missing_values[missing_values > 0].sort_values(ascending=False)
full[full['BsmtFinSF1'].isnull()]
full['BsmtFinSF1'] = full['BsmtFinSF1'].fillna(0)
full['BsmtFinSF2'] = full['BsmtFinSF2'].fillna(0)
full['TotalBsmtSF'] = full['TotalBsmtSF'].fillna(0)
full['BsmtUnfSF'] = full['BsmtUnfSF'].fillna(0)
missing_values = full.isnull().sum()
missing_values[missing_values > 0].sort_values(ascending=False)
full[full['BsmtFullBath'].isnull()]
full['BsmtFullBath'] = full['BsmtFullBath'].fillna(0)
full['BsmtHalfBath'] = full['BsmtHalfBath'].fillna(0)
missing_values = full.isnull().sum()
missing_values[missing_values > 0].sort_values(ascending=False)
full[full['Functional'].isnull()]
full[full['MasVnrArea'].isnull()]
full['MasVnrArea'] = full['MasVnrArea'].fillna(0)
missing_values = full.isnull().sum()
missing_values[missing_values > 0].sort_values(ascending=False)
full[full['GarageYrBlt'].isnull()]
null_df = full[full['GarageYrBlt'].isnull()]
null_df['GarageAge'] = null_df['YearBuilt'].apply(lambda x: 2019 - x)
null_df['GarageYrBlt'] = null_df['YearBuilt']
full[full['GarageYrBlt'].isnull()] = null_df
missing_values = full.isnull().sum()
missing_values[missing_values > 0].sort_values(ascending=False)
full[full['LotFrontage'].isnull()]
full['Condition1'] = full['Condition1'].fillna(0)
full['Condition2'] = full['Condition2'].fillna(0)
missing_values = full.isnull().sum()
missing_values[missing_values > 0].sort_values(ascending=False)
full['Alley'] = full['Alley'].fillna(-1)
missing_values = full.isnull().sum()
missing_values[missing_values > 0].sort_values(ascending=False)
full['LogLotArea'] = np.log(full['LotArea'])
full['LogLotFrontage'] = np.log(full['LotFrontage'])
plt.scatter(full['LogLotArea'], full['LogLotFrontage'])
missing_mask = full.isnull()['LotFrontage']
filled_mask = full.notnull()['LotFrontage']
predict = full[missing_mask]
train = full[filled_mask]
X_train = train['LogLotArea']
y_train = train['LogLotFrontage']
X_predict = predict['LogLotArea']
model = XGBRegressor(n_jobs=-1)