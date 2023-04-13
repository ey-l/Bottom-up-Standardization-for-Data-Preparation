import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer, FunctionTransformer, QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, learning_curve, RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, StackingRegressor, BaggingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso, SGDRegressor
set_config(display='diagram')
import scipy.stats as stats
import statsmodels.formula.api as smf
from statsmodels.graphics.gofplots import qqplot
import xgboost as xgb
from xgboost import XGBRegressor, cv as XGBcv
from lightgbm import LGBMRegressor, cv as LGBMcv
from catboost import CatBoostRegressor, cv as CBcv
rand_seed = 123
enable_graphs = 1

def plot_corr_heatmap(df, mtd='pearson', size=(50, 50), n_important=10):
    if enable_graphs == 0:
        print('Heatmap Graph disabled')
        return
    important_cols = df.corr(method=mtd).nlargest(n_important, 'SalePrice')['SalePrice'].index
    (fig, ax) = plt.subplots(figsize=size)
    sns.set(font_scale=6)
    sns.heatmap(data=df[important_cols].corr(method=mtd), annot=True, annot_kws={'fontsize': 10}, linewidths=0.7, linecolor='black', square=True, ax=ax, cbar=False, fmt='.3f', cmap='coolwarm', robust=True)
    sns.reset_orig()
    return

def plot_skewness(df, figsize=(30, 8), title=''):
    if enable_graphs == 0:
        print('Skewness Graph disabled')
        return
    ax = df.skew().plot(kind='bar', figsize=figsize, xlabel='Variables', ylabel='Skewness', title=title, grid=False, rot=90)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=5)

def normality_test(df):
    normality_test = pd.DataFrame(data={'Feature': df.columns, 'Shapiro Statistic': '', 'Shapiro p-value': '', 'Anderson-Darling Statistic': '', 'Anderson-Darling Critical Value (1%)': ''})
    for feature in normality_test['Feature']:
        normality_test['Shapiro Statistic'] = stats.shapiro(df[feature])[0]
        normality_test['Shapiro p-value'] = stats.shapiro(df[feature])[1]
        normality_test['Anderson-Darling Statistic'] = stats.anderson(df[feature])[0]
        normality_test['Anderson-Darling Critical Value (1%)'] = stats.anderson(df[feature])[1][4]
    return normality_test

def average_predictions(preds1, preds2, mtd='log'):
    if mtd == 'log':
        avrg = np.exp((np.log(preds1) + np.log(preds2)) / 2)
    else:
        avrg = np.average([preds1, preds2])
    return avrg
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1
_input0
(_input1.shape, _input0.shape)
mask = _input1.isna().mean().sort_values() > 0
ax = (_input1.loc[:, mask[mask == True].keys()].isna().mean().sort_values() * 100).plot(kind='barh', figsize=(30, 12), xlabel='Variables', title='% of missing values in Train DF')
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f%%', padding=5)
mask = _input0.isna().mean().sort_values() > 0
ax = (_input0.loc[:, mask[mask == True].keys()].isna().mean().sort_values() * 100).plot(kind='barh', figsize=(30, 15), xlabel='Variables', title='% of missing values in Test DF', grid=False)
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f%%', padding=5)
missing = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
nan_mask = _input1[_input1.columns[_input1.dtypes == object]].isna().sum().sort_values(ascending=False) > 0
_input1.loc[:, nan_mask[nan_mask == True].keys()].dtypes == object
_input1.info()
_input1.describe(include='all')
num_mask = _input1.dtypes != object
num_cols = _input1.loc[:, num_mask[num_mask == True].keys()]
num_cols.hist(figsize=(30, 15), layout=(4, 10))
normality_test(num_cols)
plot_skewness(num_cols, title='Skewness of numerical features')
num_cols.plot(kind='box', subplots=True, layout=(4, 10), sharex=False, sharey=False, figsize=(30, 15), grid=True)
plot_corr_heatmap(num_cols, size=(12, 12), n_important=15)
plot_corr_heatmap(num_cols, size=(12, 12), mtd='spearman', n_important=15)
most_corr_features = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath']
sns.pairplot(_input1[most_corr_features])
X = _input1.drop(columns=['Id', 'SalePrice'], axis=1)
y = _input1.loc[:, 'SalePrice'].to_frame()
X = X.astype(dtype={'YrSold': object, 'YearBuilt': object, 'YearRemodAdd': object, 'MoSold': object, 'GarageYrBlt': object})
areas = X.loc[:, ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'OpenPorchSF', 'PoolArea']]
sns.scatterplot(x=np.sum(areas.loc[:, ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF']], axis=1), y=areas.loc[:, 'TotalBsmtSF'])