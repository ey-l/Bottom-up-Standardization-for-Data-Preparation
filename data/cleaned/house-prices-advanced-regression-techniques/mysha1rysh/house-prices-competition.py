import os
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pandas.api.types import CategoricalDtype
from category_encoders import MEstimateEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor
plt.style.use('seaborn-whitegrid')
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large', titleweight='bold', titlesize=14, titlepad=10)
warnings.filterwarnings('ignore')

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
data_dir = Path('_data/input/house-prices-advanced-regression-techniques/')
df = pd.read_csv(data_dir / 'train.csv', index_col='Id')
df.Exterior2nd.unique()

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
(df_train, df_test) = load_data()





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
    from matplotlib.pyplot import figure
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores, height=5)
    plt.yticks(width, ticks)
    plt.title('Mutual Information Scores')
    plt.rcParams['figure.figsize'] = (12, 15)
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
plot_mi_scores(mi_scores)

def label_encode(df):
    X = df.copy()
    for colname in X.select_dtypes(['category']):
        X[colname] = X[colname].cat.codes
    return X

def mathematical_transforms(df):
    X = pd.DataFrame()
    X['LivLotRatio'] = df.GrLivArea / df.LotArea
    X['Spaciousness'] = (df.FirstFlrSF + df.SecondFlrSF) / df.TotRmsAbvGrd
    return X

def interactions(df):
    X = pd.get_dummies(df.BldgType, prefix='Bldg')
    X = X.mul(df.GrLivArea, axis=0)
    return X

def counts(df):
    X = pd.DataFrame()
    X['PorchTypes'] = df[['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'Threeseasonporch', 'ScreenPorch']].gt(0.0).sum(axis=1)
    return X

def break_down(df):
    X = pd.DataFrame()
    X['MSClass'] = df.MSSubClass.str.split('_', n=1, expand=True)[0]
    return X

def group_transforms(df):
    X = pd.DataFrame()
    X['MedNhbdArea'] = df.groupby('Neighborhood')['GrLivArea'].transform('median')
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
    kmeans = KMeans(n_clusters=20, n_init=50, random_state=0)
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

def corrplot(df, method='pearson', annot=True, **kwargs):
    sns.clustermap(df.corr(method), vmin=-1.0, vmax=1.0, cmap='icefire', method='complete', annot=annot, **kwargs)
corrplot(df_train, annot=None)

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