import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
import missingno as msno

from scipy import stats
from sklearn import preprocessing
from sklearn import feature_selection
import warnings
warnings.filterwarnings('ignore')
SEED = 42
from category_encoders import MEstimateEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor

def concat_df(train_data, test_data):
    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)

def divide_df(all_data):
    return (all_data.loc[:1459], all_data.loc[1460:])
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
y_train = df_train.SalePrice
id_val = df_train.Id
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
df_all = concat_df(df_train, df_test).drop(['SalePrice', 'Id'], axis=1)
df_train.name = 'Training Set'
df_test.name = 'Test Set'
df_all.name = 'All Set'
dfs = [df_train, df_test]
print(f'Number of Training Examples = {df_train.shape[0]}')
print(f'Number of Test Examples = {df_test.shape[0]}\n')
print(f'Training X Shape = {df_train.shape}')
print(f"Training y Shape = {df_train['SalePrice'].shape[0]}\n")
print(f'Test X Shape = {df_test.shape}')
print(f'Test y Shape = {df_test.shape[0]}\n')
print(df_train.columns)

def score_dataset(X, y, model=XGBRegressor(random_state=SEED)):
    for colname in X.select_dtypes(['category']):
        X[colname] = X[colname].cat.codes
    log_y = np.log(y)
    score = cross_val_score(model, X, log_y, cv=5, scoring='neg_mean_squared_error')
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score
null_features = df_all.isnull().sum()
null_100 = df_all.columns[list((null_features < 100) & (null_features != 0))]
num = df_all[null_100].select_dtypes(include=np.number).columns
non_num = df_all[null_100].select_dtypes(include='object').columns
df_all[num] = df_all[num].apply(lambda x: x.fillna(x.median()))
df_all[non_num] = df_all[non_num].apply(lambda x: x.fillna(x.value_counts().index[0]))
null_1000 = df_all.columns[list(null_features > 1000)]
df_all.drop(null_1000, axis=1, inplace=True)
df_all.drop(['GarageYrBlt', 'LotFrontage'], axis=1, inplace=True)
df_all['GarageCond'] = df_all['GarageCond'].fillna('Null')
df_all['GarageFinish'] = df_all['GarageFinish'].fillna('Null')
df_all['GarageQual'] = df_all['GarageQual'].fillna('Null')
df_all['GarageType'] = df_all['GarageType'].fillna('Null')
(df_train, df_test) = divide_df(df_all)
df_train = pd.concat([df_train, y_train], axis=1)
print('If the result is zero means not exist any missing values in dataset')
print(df_all.isnull().any().sum())
qualFeatures = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'GarageQual', 'GarageCond']
QualityMapping = {'Ex': 5, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0, 'NA': 0, 'Null': 0}
for feat in qualFeatures:
    df_all[feat] = df_all[feat].map(QualityMapping)

def make_categorical(df, feature):
    df[feature] = pd.Categorical(df[feature])
    df[feature] = df[feature].cat.codes
for col in df_all.select_dtypes(include=['object', 'category']).columns:
    make_categorical(df_all, col)
(df_train, df_test) = divide_df(df_all)
score_dataset(df_train, y_train)

def clean_year_sold(row):
    if row.YearBuilt > row.YrSold:
        row.YrSold = row.YearBuilt
    if row.YearRemodAdd > row.YrSold:
        row.YrSold = row.YearRemodAdd
    return row
df_all = df_all.apply(clean_year_sold, axis=1)
(df_train, df_test) = divide_df(df_all)
score_dataset(df_train, y_train)
df_all['AgeWhenSold'] = df_all['YrSold'] - df_all['YearBuilt']
(df_train, df_test) = divide_df(df_all)
score_dataset(df_train, y_train)
df_all['TotalSF'] = df_all['BsmtFinSF1'] + df_all['BsmtFinSF2'] + df_all['1stFlrSF'] + df_all['2ndFlrSF']
df_all['TotalBath'] = df_all['FullBath'] + 0.5 * df_all['HalfBath'] + df_all['BsmtFullBath'] + 0.5 * df_all['BsmtHalfBath']
df_all['TotalBsmtbath'] = df_all['BsmtFullBath'] + 0.5 * df_all['BsmtHalfBath']
df_all['TotalPorchSF'] = df_all['OpenPorchSF'] + df_all['3SsnPorch'] + df_all['EnclosedPorch'] + df_all['ScreenPorch'] + df_all['WoodDeckSF']
df_all['IsRemodel'] = df_all[['YearBuilt', 'YearRemodAdd']].apply(lambda x: 1 if x[0] != x[1] else 0, axis=1)
df_all['HasPool'] = df_all['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
df_all['Has2ndFloor'] = df_all['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
df_all['HasGarage'] = df_all['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
df_all['HasBsmt'] = df_all['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
df_all['HasFireplace'] = df_all['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
(df_train, df_test) = divide_df(df_all)
score_dataset(df_train, y_train)
df_all['LivLotRatio'] = df_all['GrLivArea'] / df_all['LotArea']
df_all['Spaciousness'] = (df_all['1stFlrSF'] + df_all['2ndFlrSF']) / df_all['TotRmsAbvGrd']
X_2 = pd.get_dummies(df_all['BldgType'], prefix='Bldg')
X_2 = X_2.mul(df_all['GrLivArea'], axis=0)
df_all = pd.concat([X_2, df_all], axis=1)
df_all['PorchTypes'] = df_all[['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']].gt(0).sum(axis=1)
df_all['MedNhbdArea'] = df_all.groupby('Neighborhood')['GrLivArea'].transform('median')
(df_train, df_test) = divide_df(df_all)
score_dataset(df_train, y_train)
qualFeatures = qualFeatures + ['OverallQual', 'OverallCond']
df_all['QualitySum'] = df_all[qualFeatures].sum(axis=1)
(df_train, df_test) = divide_df(df_all)
score_dataset(df_train, y_train)
y_SqFeetPrice = y_train / df_train['TotalSF']
encoder = MEstimateEncoder(cols=['Neighborhood'], m=5.0)