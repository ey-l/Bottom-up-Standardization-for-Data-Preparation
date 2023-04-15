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
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df_train.head()
df_train.drop('Id', axis=1, inplace=True)

def visualNA(df, perc=0):
    NAN = [(c, df[c].isna().mean() * 100) for c in df]
    NAN = pd.DataFrame(NAN, columns=['column_name', 'percentage'])
    NAN = NAN[NAN.percentage > perc]
    print(NAN.sort_values('percentage', ascending=False))
visualNA(df_train)

def handleNA(df):
    df['Alley'].fillna(value='No alley access', inplace=True)
    df['BsmtQual'].fillna(value='No Basement', inplace=True)
    df['BsmtCond'].fillna(value='No Basement', inplace=True)
    df['BsmtExposure'].fillna(value='No Basement', inplace=True)
    df['BsmtFinType1'].fillna(value='No Basement', inplace=True)
    df['BsmtFinType2'].fillna(value='No Basement', inplace=True)
    df['FireplaceQu'].fillna(value='No Fireplace', inplace=True)
    df['GarageType'].fillna(value='No Garage', inplace=True)
    df['GarageYrBlt'].fillna(value=0, inplace=True)
    df['GarageFinish'].fillna(value='No Garage', inplace=True)
    df['GarageQual'].fillna(value='No Garage', inplace=True)
    df['GarageCond'].fillna(value='No Garage', inplace=True)
    df['MasVnrType'].fillna(value='None', inplace=True)
    df['MasVnrArea'].fillna(value=0.0, inplace=True)
    df['PoolQC'].fillna(value='No Pool', inplace=True)
    df['Fence'].fillna(value='No Fence', inplace=True)
    df['MiscFeature'].fillna(value='None', inplace=True)
handleNA(df_train)
visualNA(df_train)
df_train[df_train['Electrical'].isnull()]
df_train[df_train['Electrical'].notnull() & (df_train['MSSubClass'] == 80) & (df_train['LotFrontage'] == 73)]['Electrical']
df_train.loc[df_train['Electrical'].isnull(), 'Electrical'] = 'SBrkr'
dict_neighbor = {'NAmes': {'lat': 42.04583, 'lon': -93.620767}, 'CollgCr': {'lat': 42.018773, 'lon': -93.685543}, 'OldTown': {'lat': 42.030152, 'lon': -93.614628}, 'Edwards': {'lat': 42.021756, 'lon': -93.670324}, 'Somerst': {'lat': 42.050913, 'lon': -93.644629}, 'Gilbert': {'lat': 42.060214, 'lon': -93.643179}, 'NridgHt': {'lat': 42.060357, 'lon': -93.655263}, 'Sawyer': {'lat': 42.034446, 'lon': -93.66633}, 'NWAmes': {'lat': 42.049381, 'lon': -93.634993}, 'SawyerW': {'lat': 42.033494, 'lon': -93.684085}, 'BrkSide': {'lat': 42.032422, 'lon': -93.626037}, 'Crawfor': {'lat': 42.015189, 'lon': -93.64425}, 'Mitchel': {'lat': 41.990123, 'lon': -93.600964}, 'NoRidge': {'lat': 42.051748, 'lon': -93.653524}, 'Timber': {'lat': 41.998656, 'lon': -93.652534}, 'IDOTRR': {'lat': 42.022012, 'lon': -93.622183}, 'ClearCr': {'lat': 42.060021, 'lon': -93.629193}, 'StoneBr': {'lat': 42.060227, 'lon': -93.633546}, 'SWISU': {'lat': 42.022646, 'lon': -93.644853}, 'MeadowV': {'lat': 41.991846, 'lon': -93.60346}, 'Blmngtn': {'lat': 42.059811, 'lon': -93.63899}, 'BrDale': {'lat': 42.052792, 'lon': -93.62882}, 'Veenker': {'lat': 42.040898, 'lon': -93.651502}, 'NPkVill': {'lat': 42.049912, 'lon': -93.626546}, 'Blueste': {'lat': 42.010098, 'lon': -93.647269}}
df_train['Lat'] = df_train['Neighborhood'].map(lambda neighbor: dict_neighbor[neighbor]['lat'])
df_train['Lon'] = df_train['Neighborhood'].map(lambda neighbor: dict_neighbor[neighbor]['lon'])
Categorical_features = df_train.select_dtypes(include=['object'])
Numerical_features = df_train.select_dtypes(exclude=['object'])
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
    sns.violinplot(x=feature, y='SalePrice', data=df_train)
    plt.tight_layout()

X = df_train.drop('SalePrice', axis=1)
y = df_train['SalePrice']
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