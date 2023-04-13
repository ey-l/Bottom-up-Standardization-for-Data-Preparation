import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import re
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', na_filter=False)
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', na_filter=False)
corr = _input1.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
(f, ax) = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, cmap=cmap, mask=mask, vmax=0.3, center=0, annot=False, square=True, linewidths=0.5, cbar_kws={'shrink': 0.5})
_input1[_input1.columns[1:]].corr()['SalePrice'][:].abs().sort_values(ascending=False)
df_train_20 = _input1[['Id', 'SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd']]
df_test_20 = _input0[['Id', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd']]
df_train_20
print('df_train:')
print(df_train_20.isnull().sum())
print('\ndf_test:')
print(df_test_20.isnull().sum())
X_train = df_train_20.drop(columns=['SalePrice', 'Id'])
y_train = df_train_20['SalePrice']
X_test = df_test_20.drop(columns=['Id'])
X_test = X_test.fillna(0, inplace=False)
_input1['Neighborhood'].value_counts()
dict_neighbor = {'NAmes': {'lat': 42.04583, 'lon': -93.620767}, 'CollgCr': {'lat': 42.018773, 'lon': -93.685543}, 'OldTown': {'lat': 42.030152, 'lon': -93.614628}, 'Edwards': {'lat': 42.021756, 'lon': -93.670324}, 'Somerst': {'lat': 42.050913, 'lon': -93.644629}, 'Gilbert': {'lat': 42.060214, 'lon': -93.643179}, 'NridgHt': {'lat': 42.060357, 'lon': -93.655263}, 'Sawyer': {'lat': 42.034446, 'lon': -93.66633}, 'NWAmes': {'lat': 42.049381, 'lon': -93.634993}, 'SawyerW': {'lat': 42.033494, 'lon': -93.684085}, 'BrkSide': {'lat': 42.032422, 'lon': -93.626037}, 'Crawfor': {'lat': 42.015189, 'lon': -93.64425}, 'Mitchel': {'lat': 41.990123, 'lon': -93.600964}, 'NoRidge': {'lat': 42.051748, 'lon': -93.653524}, 'Timber': {'lat': 41.998656, 'lon': -93.652534}, 'IDOTRR': {'lat': 42.022012, 'lon': -93.622183}, 'ClearCr': {'lat': 42.060021, 'lon': -93.629193}, 'StoneBr': {'lat': 42.060227, 'lon': -93.633546}, 'SWISU': {'lat': 42.022646, 'lon': -93.644853}, 'MeadowV': {'lat': 41.991846, 'lon': -93.60346}, 'Blmngtn': {'lat': 42.059811, 'lon': -93.63899}, 'BrDale': {'lat': 42.052792, 'lon': -93.62882}, 'Veenker': {'lat': 42.040898, 'lon': -93.651502}, 'NPkVill': {'lat': 42.049912, 'lon': -93.626546}, 'Blueste': {'lat': 42.010098, 'lon': -93.647269}}
_input1['Lat'] = _input1['Neighborhood'].map(lambda neighbor: dict_neighbor[neighbor]['lat'])
_input1['Lon'] = _input1['Neighborhood'].map(lambda neighbor: dict_neighbor[neighbor]['lon'])
_input0['Lat'] = _input0['Neighborhood'].map(lambda neighbor: dict_neighbor[neighbor]['lat'])
_input0['Lon'] = _input0['Neighborhood'].map(lambda neighbor: dict_neighbor[neighbor]['lon'])
_input1.select_dtypes('object').columns
from sklearn import preprocessing
for columns in _input1.select_dtypes('object').columns:
    enc = preprocessing.LabelEncoder()