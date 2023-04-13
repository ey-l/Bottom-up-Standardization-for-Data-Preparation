import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn import metrics
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1.head()
_input1 = _input1.drop('Id', axis=1, inplace=False)

def visualNA(df, perc=0):
    NAN = [(c, df[c].isna().mean() * 100) for c in df]
    NAN = pd.DataFrame(NAN, columns=['column_name', 'percentage'])
    NAN = NAN[NAN.percentage > perc]
    print(NAN.sort_values('percentage', ascending=False))
visualNA(_input1)

def handleNA(df):
    df['Alley'] = df['Alley'].fillna(value='No alley access', inplace=False)
    df['BsmtQual'] = df['BsmtQual'].fillna(value='No Basement', inplace=False)
    df['BsmtCond'] = df['BsmtCond'].fillna(value='No Basement', inplace=False)
    df['BsmtExposure'] = df['BsmtExposure'].fillna(value='No Basement', inplace=False)
    df['BsmtFinType1'] = df['BsmtFinType1'].fillna(value='No Basement', inplace=False)
    df['BsmtFinType2'] = df['BsmtFinType2'].fillna(value='No Basement', inplace=False)
    df['FireplaceQu'] = df['FireplaceQu'].fillna(value='No Fireplace', inplace=False)
    df['GarageType'] = df['GarageType'].fillna(value='No Garage', inplace=False)
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(value=0, inplace=False)
    df['GarageFinish'] = df['GarageFinish'].fillna(value='No Garage', inplace=False)
    df['GarageQual'] = df['GarageQual'].fillna(value='No Garage', inplace=False)
    df['GarageCond'] = df['GarageCond'].fillna(value='No Garage', inplace=False)
    df['MasVnrType'] = df['MasVnrType'].fillna(value='None', inplace=False)
    df['MasVnrArea'] = df['MasVnrArea'].fillna(value=0.0, inplace=False)
    df['PoolQC'] = df['PoolQC'].fillna(value='No Pool', inplace=False)
    df['Fence'] = df['Fence'].fillna(value='No Fence', inplace=False)
    df['MiscFeature'] = df['MiscFeature'].fillna(value='None', inplace=False)
handleNA(_input1)
visualNA(_input1)
_input1[_input1['Electrical'].isnull()]
_input1[_input1['Electrical'].notnull() & (_input1['MSSubClass'] == 80) & (_input1['LotFrontage'] == 73)]['Electrical']
_input1.loc[_input1['Electrical'].isnull(), 'Electrical'] = 'SBrkr'
dict_neighbor = {'NAmes': {'lat': 42.04583, 'lon': -93.620767}, 'CollgCr': {'lat': 42.018773, 'lon': -93.685543}, 'OldTown': {'lat': 42.030152, 'lon': -93.614628}, 'Edwards': {'lat': 42.021756, 'lon': -93.670324}, 'Somerst': {'lat': 42.050913, 'lon': -93.644629}, 'Gilbert': {'lat': 42.060214, 'lon': -93.643179}, 'NridgHt': {'lat': 42.060357, 'lon': -93.655263}, 'Sawyer': {'lat': 42.034446, 'lon': -93.66633}, 'NWAmes': {'lat': 42.049381, 'lon': -93.634993}, 'SawyerW': {'lat': 42.033494, 'lon': -93.684085}, 'BrkSide': {'lat': 42.032422, 'lon': -93.626037}, 'Crawfor': {'lat': 42.015189, 'lon': -93.64425}, 'Mitchel': {'lat': 41.990123, 'lon': -93.600964}, 'NoRidge': {'lat': 42.051748, 'lon': -93.653524}, 'Timber': {'lat': 41.998656, 'lon': -93.652534}, 'IDOTRR': {'lat': 42.022012, 'lon': -93.622183}, 'ClearCr': {'lat': 42.060021, 'lon': -93.629193}, 'StoneBr': {'lat': 42.060227, 'lon': -93.633546}, 'SWISU': {'lat': 42.022646, 'lon': -93.644853}, 'MeadowV': {'lat': 41.991846, 'lon': -93.60346}, 'Blmngtn': {'lat': 42.059811, 'lon': -93.63899}, 'BrDale': {'lat': 42.052792, 'lon': -93.62882}, 'Veenker': {'lat': 42.040898, 'lon': -93.651502}, 'NPkVill': {'lat': 42.049912, 'lon': -93.626546}, 'Blueste': {'lat': 42.010098, 'lon': -93.647269}}
_input1['Lat'] = _input1['Neighborhood'].map(lambda neighbor: dict_neighbor[neighbor]['lat'])
_input1['Lon'] = _input1['Neighborhood'].map(lambda neighbor: dict_neighbor[neighbor]['lon'])
Categorical_features = _input1.select_dtypes(include=['object'])
Numerical_features = _input1.select_dtypes(exclude=['object'])
Numerical_features.columns

def plotdist(data):
    try:
        sns.distplot(data)
    except RuntimeError as re:
        if str(re).startswith('Selected KDE bandwidth is 0. Cannot estimate density.'):
            sns.distplot(data, kde_kws={'bw': 0.1})
        else:
            raise re
for feature in Numerical_features.columns:
    plotdist(Numerical_features[feature])
for feature in Categorical_features.columns:
    if Categorical_features[feature].nunique() > 12:
        continue
    if Categorical_features[feature].nunique() > 5:
        plt.figure(figsize=(10, 6))
        plt.xticks(rotation=45)
    sns.violinplot(x=feature, y='SalePrice', data=_input1)
    plt.tight_layout()
X = _input1.drop('SalePrice', axis=1)
y = _input1['SalePrice']
LotFrontageX = X['LotFrontage']
GarageYrBltX = X['GarageYrBlt']
ohe = OneHotEncoder(sparse=False, drop='if_binary')
Categorical_Encoded = ohe.fit_transform(Categorical_features.astype(str))
Categorical_Encoded_Frame = pd.DataFrame(Categorical_Encoded, columns=ohe.get_feature_names(Categorical_features.columns))
Categorical_Encoded_Frame.head()
Numerical_features_X = Numerical_features.drop(['SalePrice', 'LotFrontage', 'GarageYrBlt'], axis=1)
X = Categorical_Encoded_Frame.join(Numerical_features_X).join(LotFrontageX).join(GarageYrBltX)
Xcolumns = X.columns
X.head()
X = KNNImputer(n_neighbors=5).fit_transform(X)
X = pd.DataFrame(X, columns=Xcolumns)
X.head()
sns.distplot(y)
y.skew()
y_log = np.log(y)
y_log.skew()
sns.distplot(y_log)
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y_log, test_size=0.3, random_state=42)
print('X_train Shape: ', X_train.shape)
print('X_test Shape: ', X_test.shape)
print('y_train Shape: ', y_train.shape)
print('y_test Shape: ', y_test.shape)
xStandardScaler = StandardScaler()
yStandardScaler = StandardScaler()
X_train = xStandardScaler.fit_transform(X_train)
X_test = xStandardScaler.transform(X_test)
y_train = yStandardScaler.fit_transform(y_train.ravel().reshape(-1, 1))
y_test = yStandardScaler.transform(y_test.ravel().reshape(-1, 1))
ModelCompList = list()
alphas = np.linspace(0, 0.1, num=21)
lgLasso = LassoCV(cv=10, alphas=alphas)