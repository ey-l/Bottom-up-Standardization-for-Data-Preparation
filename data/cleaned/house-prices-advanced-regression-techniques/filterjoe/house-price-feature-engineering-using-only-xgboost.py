import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pandas.api.types import CategoricalDtype
from category_encoders import MEstimateEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor

def preprocess_data(verbose=False):
    if verbose:
        print('*** BEGIN PREPROCESSING ***')
    data_directory = Path('_data/input/house-prices-advanced-regression-techniques/')
    (df, train_indices, test_indices) = load(data_directory, verbose)
    df = clean(df, verbose)
    df = order_ordinals(df, verbose)
    df = encode_nominative_categoricals(df, verbose)
    df = impute(df, verbose)
    df_train = df.loc[train_indices, :]
    df_test = df.loc[test_indices, :]
    if verbose:
        print('*** END PREPROCESSING ***\n')
    return (df_train, df_test)

def load(data_dir, verbose):
    df_train = pd.read_csv(data_dir / 'train.csv', index_col='Id')
    df_test = pd.read_csv(data_dir / 'test.csv', index_col='Id')
    df = pd.concat([df_train, df_test])
    if verbose:
        print('Train and test splits read from CSV and merged.')
        print('Missing values before any preprocessing: ', df.isnull().values.sum())
    return (df, df_train.index, df_test.index)

def clean(df, verbose):
    cleaned_features = ['Exterior2nd', 'GarageYrBlt']
    df['Exterior2nd'] = df['Exterior2nd'].replace({'Brk Cmn': 'BrkComm'})
    df['Exterior2nd'] = df['Exterior2nd'].replace({'Wd Shng': 'WdShing'})
    df['Exterior2nd'] = df['Exterior2nd'].replace({'CmentBd': 'CemntBd'})
    df['GarageYrBlt'] = df['GarageYrBlt'].where(df.GarageYrBlt <= 2010, df.YearBuilt)
    name_pairs = {'1stFlrSF': 'FirstFlrSF', '2ndFlrSF': 'SecondFlrSF', '3SsnPorch': 'ThreeSeaPorch'}
    df.rename(columns=name_pairs, inplace=True)
    if verbose:
        print('Cleaned: ', cleaned_features, sep='\n    ')
        print('Renamed (From, To): ', *name_pairs.items(), sep='\n    ')
    return df
five_levels = ['Po', 'Fa', 'TA', 'Gd', 'Ex']
ten_levels = list(range(10))
ordered_levels = {'OverallQual': ten_levels, 'OverallCond': ten_levels, 'ExterQual': five_levels, 'ExterCond': five_levels, 'BsmtQual': five_levels, 'BsmtCond': five_levels, 'HeatingQC': five_levels, 'KitchenQual': five_levels, 'FireplaceQu': five_levels, 'GarageQual': five_levels, 'GarageCond': five_levels, 'PoolQC': five_levels, 'LotShape': ['Reg', 'IR1', 'IR2', 'IR3'], 'LandSlope': ['Sev', 'Mod', 'Gtl'], 'BsmtExposure': ['No', 'Mn', 'Av', 'Gd'], 'BsmtFinType1': ['Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'], 'BsmtFinType2': ['Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'], 'Functional': ['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'], 'GarageFinish': ['Unf', 'RFn', 'Fin'], 'PavedDrive': ['N', 'P', 'Y'], 'Utilities': ['NoSeWa', 'NoSewr', 'AllPub'], 'CentralAir': ['N', 'Y'], 'Electrical': ['Mix', 'FuseP', 'FuseF', 'FuseA', 'SBrkr'], 'Fence': ['MnPrv', 'MnWw', 'GdWo', 'GdPrv']}
ordered_levels = {key: ['None'] + value for (key, value) in ordered_levels.items()}

def print_n_per_line(items, n):
    for (idx, item) in enumerate(items):
        print('    ' + item + ''.join([' ' for i in range(14 - len(item))]), end='')
        if idx % n == n - 1:
            print()
    print()

def order_ordinals(df, verbose):
    for (name, levels) in ordered_levels.items():
        df[name] = df[name].astype(CategoricalDtype(levels, ordered=True))
    if verbose:
        print('Created ordinal categories and ordered with levels: ')
        print_n_per_line(list(ordered_levels.keys()), 5)
    return df
features_nom = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'CentralAir', 'GarageType', 'MiscFeature', 'SaleType', 'SaleCondition']

def encode_nominative_categoricals(df, verbose):
    for name in features_nom:
        df[name] = df[name].astype('category')
        if 'None' not in df[name].cat.categories:
            df[name].cat.add_categories('None', inplace=True)
    if verbose:
        print('Created nominative categories (mostly conversions from numericals): ')
        print_n_per_line(features_nom, 5)
    return df

def impute(df, verbose):
    if verbose:
        print('\nMissing values before imputing:')
        sums_df = pd.DataFrame(df.isnull().sum(), columns=['Missing Values'])
        print(sums_df[sums_df['Missing Values'] > 0])
        total_missing_values = df.isnull().values.sum()
        print('Total missing values: ', total_missing_values)
    df['MSZoning'] = df.groupby('Neighborhood')['MSZoning'].transform(lambda x: x.fillna(x.mode()))
    df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])
    df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])
    df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])
    for name in df.select_dtypes('number'):
        df[name] = df[name].fillna(0)
    for name in df.select_dtypes('category'):
        df[name] = df[name].fillna('None')
    if verbose:
        print('\nmissing values imputed: ', total_missing_values)
        print('missing values remaining after imputing: ', df.isnull().values.sum())
    return df
(df_train, df_test) = preprocess_data(verbose=False)





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
print(f'Baseline score: {baseline_score:.5f} RMSLE')

def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(['object', 'category']):
        (X[colname], _) = X[colname].factorize()
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name='MI Scores', index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title('Mutual Information Scores')
X = df_train.copy()
y = X.pop('SalePrice')
mi_scores = make_mi_scores(X, y)
mi_scores

def drop_uninformative(df, mi_scores, verbose=False):
    if verbose:
        print('Dropping the following features with low mi_scores:')
        print(df.loc[:, mi_scores == 0.0].columns)
    return df.loc[:, mi_scores > 0.0]
X = df_train.copy()
y = X.pop('SalePrice')
X = drop_uninformative(X, mi_scores, verbose=True)
score_dataset(X, y)

def label_encode_these(df, cols=[], verbose=False):
    X = df.copy()
    for colname in cols:
        X[colname] = X[colname].cat.codes
    if verbose:
        print(len(cols), 'categorical features label encoded:')
        print_n_per_line(cols, 5)
    return X

def one_hot_encode_except(df, label_encoded_cols=[], verbose=False):
    X = df.copy()
    remaining_col_names = []
    for col_name in list(X):
        if X[col_name].dtype.name == 'category' and col_name not in label_encoded_cols:
            remaining_col_names.append(col_name)
    X = pd.get_dummies(X, columns=remaining_col_names)
    if verbose:
        print('One Hot encoding applied to remaining', len(remaining_col_names), 'categorical features:')
        print_n_per_line(remaining_col_names, 5)
    return X

def mathematical_transforms(df, verbose=False):
    X = pd.DataFrame()
    X['LivLotRatio'] = df.GrLivArea / df.LotArea
    X['Spaciousness'] = (df.FirstFlrSF + df.SecondFlrSF) / df.TotRmsAbvGrd
    X['TotalSF'] = df.TotalBsmtSF + df.FirstFlrSF + df.SecondFlrSF
    new_features = ['LivLotRatio', 'Spaciousness', 'TotalSF']
    if verbose and new_features:
        print(len(new_features), 'new features created via mathematical transforms: ')
        print('    ', new_features)
    return X

def interactions(df, verbose=False):
    new_interactions = ['Bldg with GrLivArea']
    columns_to_interact_with_GrLivArea = ['BldgType']
    new_prefixes = ['Bldg']
    X = pd.get_dummies(df[columns_to_interact_with_GrLivArea], columns=columns_to_interact_with_GrLivArea, prefix=new_prefixes)
    X = X.mul(df.GrLivArea, axis=0)
    if verbose and new_interactions:
        print(len(new_interactions), 'new interaction features created: ')
        print('    ', new_interactions, ' :')
        for prefix in new_prefixes:
            print_n_per_line(list(X.columns[X.columns.str.startswith(prefix)]), 5)
    return X

def counts(df, verbose=False):
    X = pd.DataFrame()
    X['PorchTypes'] = df[['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'ThreeSeaPorch', 'ScreenPorch']].gt(0.0).sum(axis=1)
    new_counted_features = ['PorchTypes']
    if verbose and new_counted_features:
        print(len(new_counted_features), 'new "feature count" type features created: ')
        print('    ', new_counted_features)
    return X

def break_down(df, verbose=False):
    X = pd.DataFrame()
    X['MSClass'] = df.MSSubClass.str.split('_', n=1, expand=True)[0]
    return X

def group_transforms(df, verbose=False):
    X = pd.DataFrame()
    X['MedNeighLivArea'] = df.groupby('Neighborhood')['GrLivArea'].transform('median')
    X['MedNeighLotArea'] = df.groupby('Neighborhood')['LotArea'].transform('median')
    X['MedQualLivArea'] = df.groupby('OverallQual')['GrLivArea'].transform('median')
    new_transforms = ['MedNeighLivArea', 'MedNeighLotArea', 'MedQualLivArea']
    if verbose and new_transforms:
        print(len(new_transforms), 'new features created via groupby transform of a feature: ')
        print('    ', new_transforms)
    return X
cluster_features = ['LotArea', 'TotalBsmtSF', 'FirstFlrSF', 'SecondFlrSF', 'GrLivArea']

def cluster_labels(df, features, n_clusters=20):
    X = df.copy()
    X_scaled = X.loc[:, features]
    X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)
    kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=0)
    X_new = pd.DataFrame()
    X_new['Cluster'] = kmeans.fit_predict(X_scaled)
    return X_new

def cluster_distance(df, features, n_clusters=20):
    X = df.copy()
    X_scaled = X.loc[:, features]
    X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)
    kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=0)
    X_cd = kmeans.fit_transform(X_scaled)
    X_cd = pd.DataFrame(X_cd, columns=[f'Centroid_{i}' for i in range(X_cd.shape[1])])
    return X_cd

def apply_pca(X, standardize=True):
    if standardize:
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    pca = PCA()
    X_pca = pca.fit_transform(X)
    component_names = [f'PC{i + 1}' for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=component_names)
    loadings = pd.DataFrame(pca.components_.T, columns=component_names, index=X.columns)
    return (pca, X_pca, loadings)

def plot_variance(pca, width=8, dpi=100):
    (fig, axs) = plt.subplots(1, 2)
    n = pca.n_components_
    grid = np.arange(1, n + 1)
    evr = pca.explained_variance_ratio_
    axs[0].bar(grid, evr)
    axs[0].set(xlabel='Component', title='% Explained Variance', ylim=(0.0, 1.0))
    cv = np.cumsum(evr)
    axs[1].plot(np.r_[0, grid], np.r_[0, cv], 'o-')
    axs[1].set(xlabel='Component', title='% Cumulative Variance', ylim=(0.0, 1.0))
    fig.set(figwidth=8, dpi=100)
    return axs

def pca_inspired(df, verbose=False):
    X = pd.DataFrame()
    X['TotSqFt'] = df.GrLivArea + df.TotalBsmtSF
    X['Feature2'] = df.YearRemodAdd * df.TotalBsmtSF
    return X

def pca_components(df, features):
    X = df.loc[:, features]
    (_, X_pca, _) = apply_pca(X)
    return X_pca
pca_features = ['GarageArea', 'YearRemodAdd', 'TotalBsmtSF', 'GrLivArea']

def indicate_outliers(df, verbose=False):
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