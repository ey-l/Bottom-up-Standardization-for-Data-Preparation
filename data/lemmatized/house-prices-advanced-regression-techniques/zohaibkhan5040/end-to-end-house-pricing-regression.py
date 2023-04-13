import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
(_input1.shape, _input0.shape)
_input1.head()
_input1.describe()
_input1.info()
_input1 = _input1.drop('Id', axis=1, inplace=False)
num_columns = _input1.select_dtypes(include=np.number).columns.tolist()
num_columns.remove('SalePrice')
cat_columns = _input1.select_dtypes(exclude=np.number).columns.tolist()
len(num_columns) + len(cat_columns) + 1 == len(_input1.columns)
repetitive = ['Bsmt', 'Garage', 'Sale', 'Kitchen']
similar_cols = []
print('Looking for highly similar variable names')
print('--' * 30)
for col in num_columns + cat_columns:
    if any((x in col for x in repetitive)):
        print(col)
        similar_cols.append(col)
print('Looking at Categorical Variable Cardinalities')
print('--' * 30)
for col in cat_columns:
    uniques = _input1[col].unique()
    if len(uniques) > 10:
        print(f'{len(uniques)} values in {col}')
    else:
        print(f'{len(uniques)} values in {col}: {uniques}')
print('Checking for Low Cardinality Numeric Variables')
print('--' * 30)
for col in num_columns:
    uniques = _input1[col].unique()
    if len(uniques) < 20:
        print(f'{len(uniques)} unique values in {col}: {sorted(uniques)}')
corr_matrix = _input1.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, cmap='Blues')
target_var = 'SalePrice'
corr_matrix[target_var].apply(lambda x: abs(x)).sort_values(ascending=False)
sns.scatterplot(x='OverallQual', y='SalePrice', data=_input1)
sns.distplot(_input1[target_var])
for i in range(95, 100):
    print(f'{i}% of the target values lie under: {int(np.percentile(_input1[target_var], i))}')
print(f'Critical Values:\n\tMax:{_input1[target_var].max()}\n\tMin:{_input1[target_var].min()}')
upper_thresh = 38500
print(f'Before Log Transform: Skewness {stats.skew(_input1.SalePrice)}')
_input1['SalePrice'] = np.log1p(_input1['SalePrice'])
print(f'After Log Transform: Skewness {stats.skew(_input1.SalePrice)}')
print(f'Applying Inverse Transformation: Skewness {stats.skew(np.expm1(_input1.SalePrice))}')
print(f'Final Skewness: {stats.skew(_input1.SalePrice)}')
sns.distplot(_input1['SalePrice'])
X = _input1.drop(target_var, axis=1)
y = _input1[target_var]
(X.shape, y.shape)
missing_count = X.isnull().sum()
missing_count = missing_count[missing_count > 0]
missing_cols = pd.DataFrame(missing_count).index.tolist()
plt.figure(figsize=(12, 8))
sns.heatmap(X[missing_cols].isnull(), cmap='viridis', cbar=False)
missing_count.sort_values(ascending=False) / len(X) * 100
print(X[missing_cols].dtypes)
X[missing_cols].head(10)

def handle_missing(df):
    cols = ['LotFrontage', 'MasVnrArea']
    for col in cols:
        df[col] = df[col].fillna(df[col].median())
    none_fill_cols = 'Alley BsmtQual BsmtCond BsmtExposure BsmtFinType1 BsmtFinType2 Electrical FireplaceQu GarageType GarageFinish GarageQual GarageCond PoolQC Fence MiscFeature'.split()
    df[none_fill_cols] = df[none_fill_cols].fillna('NONE')
    df['Electrical'] = df['Electrical'].fillna('SBrkr')
    df['MasVnrType'] = df['MasVnrType'].fillna(df.MasVnrType.mode())
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(exclude=np.number).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna('NONE')
    return df
tmp = X.copy()
tmp = handle_missing(tmp)
tmp.isnull().sum()[tmp.isnull().sum() > 0]
X = handle_missing(X)
X.isnull().sum().max()

def new_features(X):
    X['HasWoodDeck'] = (X['WoodDeckSF'] == 0) * 1
    X['HasOpenPorch'] = (X['OpenPorchSF'] == 0) * 1
    X['HasEnclosedPorch'] = (X['EnclosedPorch'] == 0) * 1
    X['Has3SsnPorch'] = (X['3SsnPorch'] == 0) * 1
    X['HasScreenPorch'] = (X['ScreenPorch'] == 0) * 1
    X['Total_Home_Quality'] = X['OverallQual'] + X['OverallCond']
    X['TotalSF'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']
    X['TotalSquareFootage'] = X['BsmtFinSF1'] + X['BsmtFinSF2'] + X['1stFlrSF'] + X['2ndFlrSF']
    X['HasPool'] = X['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    X['Has2ndFloor'] = X['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    X['HasGarage'] = X['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    X['HasBsmt'] = X['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    X['HasFireplace'] = X['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
    return X
X = new_features(X)
len(X.columns)
num_columns = X.select_dtypes(include=np.number).columns
skewed_features = X[num_columns].apply(lambda x: abs(stats.skew(x))).sort_values(ascending=False)
high_skewed = skewed_features[skewed_features > 0.5]
high_skewed
(X.shape, y.shape)
from sklearn import preprocessing
cat_columns = X.select_dtypes(exclude=np.number).columns
fi_data = X.copy()
for feat in cat_columns:
    fi_data[feat] = preprocessing.LabelEncoder().fit_transform(fi_data[feat])
from sklearn.ensemble import RandomForestRegressor