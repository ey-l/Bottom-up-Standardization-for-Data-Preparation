import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, PowerTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
df_sample_sub = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df_test.head()
df_train.head()
df_train['SalePrice'].describe()
sb.distplot(df_train['SalePrice'])
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
var = 'LotArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
(f, ax) = plt.subplots(figsize=(8, 6))
fig = sb.boxplot(x=var, y='SalePrice', data=data)
fig.axis(ymin=0, ymax=800000)
var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
(f, ax) = plt.subplots(figsize=(20, 16))
fig = sb.boxplot(x=var, y='SalePrice', data=data)
fig.axis(ymin=0, ymax=800000)
sb.swarmplot(x='YearBuilt', y='SalePrice', data=df_train, color='.25')
plt.xticks(weight='bold', rotation=90)
var = 'MSZoning'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data = data.sort_values(by='SalePrice', ascending=True)
(f, ax) = plt.subplots(figsize=(8, 6))
fig = sb.boxplot(x=var, y='SalePrice', data=data)
fig.axis(ymin=0, ymax=800000)
var = 'Neighborhood'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data = data.sort_values(by='SalePrice', ascending=True)
(f, ax) = plt.subplots(figsize=(8, 6))
fig = sb.boxplot(x=var, y='SalePrice', data=data)
fig.axis(ymin=0, ymax=800000)
plt.xticks(weight='bold', rotation=90)
correlation_train = df_train.corr()
sb.set(font_scale=2)
plt.figure(figsize=(45, 70))
ax = sb.heatmap(correlation_train, annot=True, annot_kws={'size': 25}, fmt='.1f', cmap='PiYG', linewidths=0.5)
corr_dict = correlation_train['SalePrice'].sort_values(ascending=False).to_dict()
important_columns = []
for (key, value) in corr_dict.items():
    if (value >= 0.5) & (value < 0.999) | (value <= -0.5) & (value > -0.999):
        important_columns.append(key)
important_columns
sb.set()
sb.set(font_scale=2)
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd']
sb.pairplot(df_train[cols], size=3.5)

pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 81)
train_test = pd.concat([df_train, df_test], axis=0, sort=False)
train_test.head()
total = train_test.isnull().sum().sort_values(ascending=False)
percent = (train_test.isnull().sum() / train_test.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data[missing_data['Total'] > 0]
train_test.PoolQC.unique()
PoolArea = pd.DataFrame(train_test['PoolArea'], columns=['PoolArea'])
PoolQC = pd.DataFrame(train_test['PoolQC'], columns=['PoolQC'])
Df_Pool = pd.concat([PoolArea, PoolQC], axis=1)
Df_Pool[(Df_Pool['PoolArea'] != 0) & Df_Pool['PoolArea'].isna()]
Df_Pool[Df_Pool['PoolQC'].isna()].shape
Df_Pool[Df_Pool['PoolArea'] == 0].shape
train_test['PoolQC'] = train_test['PoolQC'].fillna('NA')
train_test[train_test['MiscFeature'].isna()].shape
train_test['MiscFeature'] = train_test['MiscFeature'].fillna('NA')
train_test[train_test['Alley'].isna()].shape
train_test['Alley'] = train_test['Alley'].fillna('NA')
train_test[train_test['Fence'].isna()].shape
train_test['Fence'] = train_test['Fence'].fillna('NA')
train_test.FireplaceQu.unique()
FireplaceQu = pd.DataFrame(train_test['FireplaceQu'], columns=['FireplaceQu'])
Fireplaces = pd.DataFrame(train_test['Fireplaces'], columns=['Fireplaces'])
Df_Fireplace = pd.concat([FireplaceQu, Fireplaces], axis=1)
Df_Fireplace[(Df_Fireplace['Fireplaces'] != 0) & Df_Fireplace['FireplaceQu'].isna()]
Df_Fireplace[Df_Fireplace['FireplaceQu'].isna()].shape
Df_Fireplace[Df_Fireplace['Fireplaces'] == 0].shape
train_test['FireplaceQu'][Df_Fireplace['FireplaceQu'].isna()] = train_test['FireplaceQu'][Df_Fireplace['FireplaceQu'].isna()].fillna('NA')
train_test.LotFrontage.unique()
train_test['LotFrontage'][train_test.LotFrontage == 'NA'] = 0
pd.set_option('display.max_rows', 5000)
idD = pd.DataFrame(train_test['Id'], columns=['Id'])
LotArea = pd.DataFrame(train_test['LotArea'], columns=['LotArea'])
LotFrontage = pd.DataFrame(train_test['LotFrontage'], columns=['LotFrontage'])
Neighborhood = pd.DataFrame(train_test['Neighborhood'], columns=['Neighborhood'])
Df_LotFrontage = pd.concat([LotArea, LotFrontage, Neighborhood, idD], axis=1)
Df_LotFrontage.head()
KNN = KNeighborsRegressor(n_neighbors=3)
for (name, group) in Df_LotFrontage.groupby('Neighborhood'):
    if group[group['LotFrontage'].isna() & (group['LotFrontage'] != 0)].shape[0] >= 3:
        DF_LotFrontage_train = group[group['LotFrontage'].notna() & (group['LotFrontage'] != 0)]
        X = DF_LotFrontage_train.drop(['LotFrontage', 'Id'], axis=1)
        Y = DF_LotFrontage_train['LotFrontage']
        X['Neighborhood'] = 5000000