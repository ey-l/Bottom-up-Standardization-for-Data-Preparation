import os
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pandas.api.types import CategoricalDtype
from scipy.stats import norm
from category_encoders import MEstimateEncoder
from category_encoders import CatBoostEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor
import optuna
warnings.filterwarnings('ignore')
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
df_train.head()
df_train.describe()
df_train.info()
df_test.head()
df_test.describe()
df_test.info()

def load_data():
    data_dir = Path('_data/input/house-prices-advanced-regression-techniques/')
    df_train = pd.read_csv(data_dir / 'train.csv', index_col='Id')
    df_test = pd.read_csv(data_dir / 'test.csv', index_col='Id')
    df = pd.concat([df_train, df_test])
    df = clean(df)
    df = encode(df)
    df = impute(df)
    df_train = df.loc[df_train.index, :]
    df_test = df.loc[df_test.index, :]
    return (df_train, df_test)
plt.figure(figsize=(20, 8))
sns.distplot(df_train.SalePrice, fit=norm)
num_columns = df_train.columns[df_train.dtypes != 'object']
cat_columns = df_train.columns[df_train.dtypes == 'object']
print(num_columns, num_columns.shape)
print(cat_columns, cat_columns.shape)
df = pd.concat([df_train, df_test])
num_columns = df.columns[df.dtypes != 'object']
cat_columns = df.columns[df.dtypes == 'object']
(figure1, ax) = plt.subplots(7, 6, figsize=(25, 25))
for (i, j) in enumerate(num_columns):
    sns.distplot(df[j], fit=norm, ax=ax[i // 6, i % 6])
figure1.text(0.4, 0.92, 'Distribution of numerical features', size=25, weight='bold')

(figure2, ax) = plt.subplots(8, 6, figsize=(25, 25))
for (k, j) in enumerate(cat_columns):
    plot = sns.barplot(data=pd.DataFrame(df[j].value_counts()).reset_index(), x='index', y=j, ax=ax[k // 6, k % 6])
    plot.set(title=j)
    plot.set(xticks=[])
figure2.text(0.4, 0.92, 'Distribution of categorical features', size=25, weight='bold')


def clean(df):
    df['Exterior2nd'] = df['Exterior2nd'].replace({'Brk Cmn': 'BrkComm'})
    df['GarageYrBlt'] = df['GarageYrBlt'].where(df.GarageYrBlt <= 2010, df.YearBuilt)
    df.rename(columns={'1stFlrSF': 'FirstFlrSF', '2ndFlrSF': 'SecondFlrSF', '3SsnPorch': 'Threeseasonporch'}, inplace=True)
    return df
features_nom = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'CentralAir', 'GarageType', 'MiscFeature', 'SaleType', 'SaleCondition']
five_levels = ['Po', 'Fa', 'TA', 'Gd', 'Ex']
ten_levels = list(range(10))
ordered_levels = {'OverallQual': ten_levels, 'OverallCond': ten_levels, 'ExterQual': five_levels, 'ExterCond': five_levels, 'BsmtQual': five_levels, 'BsmtCond': five_levels, 'HeatingQC': five_levels, 'KitchenQual': five_levels, 'FireplaceQu': five_levels, 'GarageQual': five_levels, 'GarageCond': five_levels, 'PoolQC': five_levels, 'LotShape': ['Reg', 'IR1', 'IR2', 'IR3'], 'LandSlope': ['Sev', 'Mod', 'Gtl'], 'BsmtExposure': ['No', 'Mn', 'Av', 'Gd'], 'BsmtFinType1': ['Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'], 'BsmtFinType2': ['Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'], 'Functional': ['Sal', 'Sev', 'Maj1', 'Maj2', 'Mod', 'Min2', 'Min1', 'Typ'], 'GarageFinish': ['Unf', 'RFn', 'Fin'], 'PavedDrive': ['N', 'P', 'Y'], 'Utilities': ['NoSeWa', 'NoSewr', 'AllPub'], 'CentralAir': ['N', 'Y'], 'Electrical': ['Mix', 'FuseP', 'FuseF', 'FuseA', 'SBrkr'], 'Fence': ['MnWw', 'GdWo', 'MnPrv', 'GdPrv']}
ordered_levels = {key: ['None'] + value for (key, value) in ordered_levels.items()}

def encode(df):
    for name in features_nom:
        df[name] = df[name].astype('category')
        if 'None' not in df[name].cat.categories:
            df[name].cat.add_categories('None', inplace=True)
    for (name, levels) in ordered_levels.items():
        df[name] = df[name].astype(CategoricalDtype(levels, ordered=True))
    return df

def impute(df):
    for name in df.select_dtypes('number'):
        df[name] = df[name].fillna(0)
    for name in df.select_dtypes('category'):
        df[name] = df[name].fillna('None')
    return df
df = clean(df)
df = encode(df)
df = impute(df)
df_train = df.loc[df_train.index, :]
df_test = df.loc[df_test.index, :]
df_train.info()
df_test.info()

def score_dataset(X, y, model=XGBRegressor()):
    for colname in X.select_dtypes(['category']):
        X[colname] = X[colname].cat.codes
    log_y = np.log(y)
    score = cross_val_score(model, X, log_y, cv=5, scoring='neg_mean_squared_error')
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score
X = df_train.copy()
y = X.pop('SalePrice')
baseline_score = score_dataset(X, y)
print(baseline_score)

def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(['object', 'category']):
        (X[colname], _) = X[colname].factorize()
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name='MI Scores', index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores
X = df_train.copy()
y = X.pop('SalePrice')
mi_scores = make_mi_scores(X, y)
mi_scores

def drop_uninformative(df, mi_scores):
    return df.loc[:, mi_scores > 0.0]
X = df_train.copy()
y = X.pop('SalePrice')
X = drop_uninformative(X, mi_scores)
score_dataset(X, y)

def label_encode(df):
    X = df.copy()
    for colname in X.select_dtypes(['category']):
        X[colname] = X[colname].cat.codes
    return X

def mathematical_transforms(df):
    X = pd.DataFrame()
    X['LivLotRatio'] = df.GrLivArea / df.LotArea
    X['Spaciousness'] = (df.FirstFlrSF + df.SecondFlrSF) / df.TotRmsAbvGrd
    X['Feet'] = np.sqrt(df.GrLivArea)
    X['TotalSF'] = df.TotalBsmtSF + df.FirstFlrSF + df.SecondFlrSF
    X['TotalBathrooms'] = df.FullBath + 0.5 * df.HalfBath + df.BsmtFullBath + 0.5 * df.BsmtHalfBath
    X['TotalPorchSF'] = df.OpenPorchSF + df.Threeseasonporch + df.EnclosedPorch + df.ScreenPorch + df.WoodDeckSF
    return X

def interactions(df):
    X = pd.get_dummies(df.BldgType, prefix='Bldg')
    X = X.mul(df.GrLivArea, axis=0)
    return X

def counts(df):
    X = pd.DataFrame()
    X['PorchTypes'] = df[['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'Threeseasonporch', 'ScreenPorch']].gt(0.0).sum(axis=1)
    return X

def group_transforms(df):
    X = pd.DataFrame()
    X['MedNhbdArea'] = df.groupby('Neighborhood')['GrLivArea'].transform('median')
    return X

def break_down(df):
    X = pd.DataFrame()
    X['MSClass'] = df.MSSubClass.str.split('_', n=1, expand=True)[0]
    return X
cluster_features = ['LotArea', 'TotalBsmtSF', 'FirstFlrSF', 'SecondFlrSF', 'GrLivArea']

def cluster_distance(df, features, n_clusters=20):
    X = df.copy()
    X_scaled = X.loc[:, features]
    X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)
    kmeans = KMeans(n_clusters=20, n_init=50, random_state=0)
    X_cd = kmeans.fit_transform(X_scaled)
    X_cd = pd.DataFrame(X_cd, columns=[f'centroid_{i}' for i in range(X_cd.shape[1])])
    return X_cd

def cluster_labels(df, features, n_clusters=20):
    X = df.copy()
    X_scaled = X.loc[:, features]
    X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)
    kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=0)
    X_new = pd.DataFrame()
    X_new['Cluster'] = kmeans.fit_predict(X_scaled)
    return X_new

def apply_pca(X, standarize=True):
    if standarize:
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    pca = PCA()
    X_pca = pca.fit_transform(X)
    component_names = [f'PC{i + 1}' for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=component_names)
    loadings = pd.DataFrame(pca.components_.T, columns=component_names, index=X.columns)
    return (pca, X_pca, loadings)

def pca_inspired(df):
    X = pd.DataFrame()
    X['Feature1'] = df.GrLivArea + df.TotalBsmtSF
    X['Feature2'] = df.YearRemodAdd * df.TotalBsmtSF
    return X

def pca_components(df, features):
    X = df.loc[:, features]
    (_, X_pca, _) = apply_pca(X)
    return X_pca
pca_features = ['GarageArea', 'YearRemodAdd', 'TotalBsmtSF', 'GrLivArea']

def indicate_outliers(df):
    X_new = pd.DataFrame()
    X_new['Outlier'] = (df.Neighborhood == 'Edwards') & (df.SaleCondition == 'Partial')
    return X_new

class CrossFoldEncoder:

    def __init__(self, encoder, **kwargs):
        self.encoder_ = encoder
        self.kwargs_ = kwargs
        self.cv_ = KFold(n_splits=5)

    def fit_transform(self, X, y, cols):
        self.fitted_encoders_ = []
        self.cols_ = cols
        X_encoded = []
        for (idx_encode, idx_train) in self.cv_.split(X):
            fitted_encoder = self.encoder_(cols=cols, **self.kwargs_)