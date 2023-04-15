import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
df_train = df_train.drop(columns='Id')
df_test = df_test.drop(columns='Id')
Check_years = df_train.columns[df_train.columns.str.contains(pat='Year|Yr')]
df_train[Check_years.values].max().sort_values(ascending=False)
df_test[Check_years.values].max().sort_values(ascending=False)
Replace_year = df_test.loc[df_test['GarageYrBlt'] > 2050, 'GarageYrBlt'].index.tolist()
df_test.loc[Replace_year, 'GarageYrBlt'] = df_test['GarageYrBlt'].mode()
train_missing = df_train.count().loc[df_train.count() < 1460].sort_values(ascending=False)
sns.set_theme(rc={'grid.linewidth': 0.6, 'grid.color': 'white', 'axes.linewidth': 1, 'axes.facecolor': '#ECECEC', 'axes.labelcolor': '#000000', 'figure.facecolor': 'white', 'xtick.color': '#000000', 'ytick.color': '#000000'})
with plt.rc_context(rc={'figure.dpi': 120, 'axes.labelsize': 8.5, 'xtick.labelsize': 6, 'ytick.labelsize': 6}):
    (fig, ax) = plt.subplots(1, 1, figsize=(6, 4))
    sns.barplot(x=train_missing.values, y=train_missing.index, palette='viridis')
    plt.xlabel('Non-Na values')

test_missing = df_test.count().loc[df_test.count() < 1459].sort_values(ascending=False)
with plt.rc_context(rc={'figure.dpi': 120, 'axes.labelsize': 8.5, 'xtick.labelsize': 6, 'ytick.labelsize': 6}):
    (fig, ax) = plt.subplots(1, 1, figsize=(7, 6))
    sns.barplot(x=test_missing.values, y=test_missing.index, palette='viridis')
    plt.xlabel('Non-Na values')

None_category = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageCond', 'GarageQual', 'GarageFinish', 'GarageType', 'BsmtCond', 'BsmtExposure', 'BsmtQual', 'BsmtFinType1', 'BsmtFinType2']
for column in None_category:
    df_train.loc[df_train[column].isnull(), column] = 'None'
    df_test.loc[df_test[column].isnull(), column] = 'None'
df_train.loc[:, df_train.isna().sum() > 0].isna().sum().sort_values(ascending=False)
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
cont_vars = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'ScreenPorch']
knn_vars_train_cont = df_train[cont_vars].copy()
Scaler = RobustScaler()
knn_vars_train_cont = pd.DataFrame(Scaler.fit_transform(knn_vars_train_cont), columns=['col' + str(i) for i in range(0, 15)])
train_imp_cont = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
train_imp_cont_results = train_imp_cont.fit_transform(knn_vars_train_cont)
train_imp_cont_results = pd.DataFrame(Scaler.inverse_transform(train_imp_cont_results), columns=['col' + str(i) for i in range(0, 15)])
df_train['LotFrontage'] = train_imp_cont_results['col0']
df_train['MasVnrArea'] = train_imp_cont_results['col2'].astype('float64')
for column in ['MasVnrType', 'Electrical']:
    df_train.loc[df_train[column].isnull(), column] = df_train[column].mode()[0]
from sklearn.preprocessing import LabelEncoder
knn_vars_train_cat = df_train.drop(cont_vars, axis=1)
knn_vars_train_cat = knn_vars_train_cat.drop('SalePrice', axis=1)
obj_vars = knn_vars_train_cat.select_dtypes(include=['object', 'category']).columns
for column in obj_vars:
    knn_vars_train_cat[column] = LabelEncoder().fit_transform(knn_vars_train_cat[column])
train_imp_cat = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
train_imp_cat_results = train_imp_cat.fit_transform(knn_vars_train_cat)
train_imp_cat_results = pd.DataFrame(train_imp_cat_results, columns=['col' + str(i) for i in range(0, 64)])
df_train['GarageYrBlt'] = train_imp_cat_results['col48']
df_train['GarageYrBlt'] = df_train['GarageYrBlt'].astype('int64')
knn_vars_test_cont = df_test[cont_vars].copy()
Scaler = StandardScaler()
knn_vars_test_cont = pd.DataFrame(Scaler.fit_transform(knn_vars_test_cont), columns=['col' + str(i) for i in range(0, 15)])
test_imp_cont = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
test_imp_cont_results = test_imp_cont.fit_transform(knn_vars_test_cont)
test_imp_cont_results = pd.DataFrame(Scaler.inverse_transform(test_imp_cont_results), columns=['col' + str(i) for i in range(0, 15)])
df_test['LotFrontage'] = test_imp_cont_results['col0']
for column in df_test.columns:
    if (df_test[column].isnull().sum() <= 60) & (df_test[column].isnull().sum() > 0) & ((df_test[column].dtypes == 'O') | (df_test[column].dtypes == 'float64')) & (df_test[column].nunique() < 20):
        df_test.loc[df_test[column].isnull(), column] = df_test[column].mode()[0]
    elif (df_test[column].isnull().sum() <= 60) & (df_test[column].isnull().sum() > 0) & (df_test[column].dtypes == 'float64') & (df_test[column].nunique() > 100):
        df_test.loc[df_test[column].isnull(), column] = df_test[column].mean()
    else:
        pass
knn_vars_test_cat = df_test.drop(cont_vars, axis=1)
for column in knn_vars_test_cat:
    knn_vars_test_cat[column] = LabelEncoder().fit_transform(knn_vars_test_cat[column])
test_imp_cat = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
test_imp_cat_results = test_imp_cat.fit_transform(knn_vars_test_cat)
test_imp_cat_results = pd.DataFrame(test_imp_cat_results, columns=['col' + str(i) for i in range(0, 64)])
df_test['GarageYrBlt'] = test_imp_cat_results['col48']
df_test['GarageYrBlt'] = df_test['GarageYrBlt'].astype('int64')
print(df_train.isna().sum().any(), df_test.isna().sum().any(), sep='\n')
train_obj = df_train.select_dtypes(include=['object', 'category']).columns
train_int_float = df_train.select_dtypes(include=['int64', 'float64'])
col_order = train_int_float.nunique().sort_values(ascending=False).index.tolist()
train_int_float = train_int_float[col_order].columns
with plt.rc_context(rc={'figure.dpi': 500, 'axes.labelsize': 7, 'xtick.labelsize': 5, 'ytick.labelsize': 5}):
    (fig, ax) = plt.subplots(5, 5, figsize=(8.5, 10), sharey=True)
    for (idx, (column, axes)) in list(enumerate(zip(train_int_float[0:22], ax.flatten()))):
        sns.scatterplot(ax=axes, x=df_train[column], y=np.log(df_train['SalePrice']), hue=np.log(df_train['SalePrice']), palette='viridis', alpha=0.7, s=8)
        axes.legend([], [], frameon=False)
    else:
        [axes.set_visible(False) for axes in ax.flatten()[idx + 1:]]
    plt.tight_layout()

with plt.rc_context(rc={'figure.dpi': 500, 'axes.labelsize': 7, 'xtick.labelsize': 5, 'ytick.labelsize': 5}):
    (fig, ax) = plt.subplots(5, 4, figsize=(8.5, 9), sharey=True)
    for (idx, (column, axes)) in list(enumerate(zip(train_int_float[22:], ax.flatten()))):
        sns.scatterplot(ax=axes, x=df_train[column], y=np.log(df_train['SalePrice']), hue=np.log(df_train['SalePrice']), palette='viridis', alpha=0.7, s=8)
        axes.legend([], [], frameon=False)
    else:
        [axes.set_visible(False) for axes in ax.flatten()[idx + 1:]]
    plt.tight_layout()

train_cont_balanced = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'ScreenPorch']
train_cont_unbalanced = ['LowQualFinSF', '3SsnPorch', 'PoolArea', 'MiscVal']
train_cat = df_train.drop(train_cont_balanced, axis=1).columns.tolist()
train_cat.remove('SalePrice')
df_train[train_cat].loc[:, df_train.nunique() > 25].nunique().sort_values(ascending=False)
train_high_cat = df_train[train_cat].loc[:, df_train.nunique() > 25].copy()
for column in train_high_cat.columns:
    train_high_cat[column] = train_high_cat[column].astype('category')
with plt.rc_context(rc={'figure.dpi': 450, 'axes.labelsize': 5, 'xtick.labelsize': 4, 'ytick.labelsize': 4}):
    (fig, ax) = plt.subplots(1, 3, figsize=(6, 7.5))
    for (idx, (column, axes)) in list(enumerate(zip(train_high_cat.columns, ax.flatten()))):
        sns.stripplot(ax=axes, x=np.log(df_train['SalePrice']), y=train_high_cat[column], palette='viridis', alpha=0.95, size=1.5)
        sns.boxplot(ax=axes, x=np.log(df_train['SalePrice']), y=train_high_cat[column], showmeans=True, meanline=True, zorder=10, meanprops={'color': 'r', 'linestyle': '-', 'lw': 0.8}, medianprops={'visible': False}, whiskerprops={'visible': False}, showfliers=False, showbox=False, showcaps=False)
        sns.pointplot(ax=axes, x=np.log(df_train['SalePrice']), y=train_high_cat[column], ci=None, color='r', scale=0.15)
    else:
        [axes.set_visible(False) for axes in ax.flatten()[idx + 1:]]
    plt.tight_layout()

train_norm_cat = df_train[train_cat].loc[:, df_train.nunique() <= 25].columns.tolist()
with plt.rc_context(rc={'figure.dpi': 500, 'axes.labelsize': 7, 'xtick.labelsize': 5.5, 'ytick.labelsize': 5.5}):
    (fig, ax) = plt.subplots(5, 3, figsize=(8, 13), sharey=True)
    for (idx, (column, axes)) in list(enumerate(zip(train_norm_cat[:15], ax.flatten()))):
        order = df_train.groupby(column)['SalePrice'].mean().sort_values(ascending=True).index
        sns.violinplot(ax=axes, x=df_train[column], y=np.log(df_train['SalePrice']), order=order, scale='width', linewidth=0.3, palette='viridis', saturation=0.5, inner=None)
        plt.setp(axes.collections, alpha=0.3)
        sns.stripplot(ax=axes, x=df_train[column], y=np.log(df_train['SalePrice']), palette='viridis', s=1.3, alpha=0.9, order=order)
        sns.boxplot(ax=axes, x=df_train[column], order=order, y=np.log(df_train['SalePrice']), showmeans=True, meanline=True, zorder=10, meanprops={'color': 'r', 'linestyle': '--', 'lw': 0.6}, medianprops={'visible': False}, whiskerprops={'visible': False}, showfliers=False, showbox=False, showcaps=False)
        if df_train[column].nunique() > 5:
            plt.setp(axes.get_xticklabels(), rotation=90)
    else:
        [axes.set_visible(False) for axes in ax.flatten()[idx + 1:]]
    plt.tight_layout()

indx_final = [30, 462, 495, 523, 588, 632, 968, 1298, 1324]
df_train = df_train.drop(indx_final, axis=0).reset_index(drop=True)
df_train['TotalPorch'] = df_train['ScreenPorch'] + df_train['EnclosedPorch'] + df_train['3SsnPorch'] + df_train['ScreenPorch']
df_train['Rooms_kitchens'] = df_train['TotRmsAbvGrd'] + df_train['BsmtFullBath'] + df_train['BsmtHalfBath'] + df_train['FullBath'] + df_train['HalfBath']
df_train['Sqr_feet_per_room'] = (df_train['1stFlrSF'] + df_train['2ndFlrSF']) / df_train['TotRmsAbvGrd']
train_cont_balanced.append('TotalPorch')
train_cont_balanced.append('Sqr_feet_per_room')
df_test['TotalPorch'] = df_test['ScreenPorch'] + df_test['EnclosedPorch'] + df_test['3SsnPorch'] + df_test['ScreenPorch']
df_test['Rooms_kitchens'] = df_test['TotRmsAbvGrd'] + df_test['BsmtFullBath'] + df_test['BsmtHalfBath'] + df_test['FullBath'] + df_test['HalfBath']
df_test['Sqr_feet_per_room'] = (df_test['1stFlrSF'] + df_test['2ndFlrSF']) / df_test['TotRmsAbvGrd']
with plt.rc_context(rc={'figure.dpi': 500, 'axes.labelsize': 7, 'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 6, 'legend.title_fontsize': 6}):
    (fig, ax) = plt.subplots(1, 4, figsize=(8, 3), sharey=True)
    for (idx, (column, axes)) in list(enumerate(zip(train_cont_unbalanced, ax.flatten()))):
        sns.scatterplot(ax=axes, x=df_train[column], y=np.log(df_train['SalePrice']), hue=np.log(df_train['SalePrice']), palette='viridis', alpha=0.8, s=9)
    axes_legend = ax.flatten()
    axes_legend[0].legend(title='SalePrice', loc='lower right')
    axes_legend[1].legend(title='SalePrice', loc='lower right')
    axes_legend[3].legend(title='SalePrice', loc='lower right')
    plt.tight_layout()

for column in train_cont_unbalanced:
    df_train.loc[df_train[column] == 0, column] = 'None'
    df_train.loc[(df_train[column] != 0) & (df_train[column] != 'None'), column] = 'Present'
for column in train_cont_unbalanced:
    df_test.loc[df_test[column] == 0, column] = 'None'
    df_test.loc[(df_test[column] != 0) & (df_test[column] != 'None'), column] = 'Present'
with plt.rc_context(rc={'figure.dpi': 500, 'axes.labelsize': 7, 'xtick.labelsize': 5, 'ytick.labelsize': 5}):
    (fig, ax) = plt.subplots(5, 4, figsize=(8.5, 9))
    for (idx, (column, axes)) in list(enumerate(zip(train_cont_balanced, ax.flatten()))):
        sns.kdeplot(ax=axes, x=df_train[column], fill=True, alpha=0.2, color='#006e7a', linewidth=0.8)
    else:
        [axes.set_visible(False) for axes in ax.flatten()[idx + 1:]]
    plt.tight_layout()

df_train[train_cont_balanced] = np.log(df_train[train_cont_balanced] + 1)
df_test[train_cont_balanced] = np.log(df_test[train_cont_balanced] + 1)
from sklearn.model_selection import KFold

def mean_encode(train_data, test_data, columns, target_col, alpha=0, folds=1):
    encoded_cols = []
    target_mean_global = train_data[target_col].mean()
    for col in columns:
        nrows_cat = train_data.groupby(col)[target_col].count()
        target_means_cats = train_data.groupby(col)[target_col].mean()
        target_means_cats_adj = (target_means_cats * nrows_cat + target_mean_global * alpha) / (nrows_cat + alpha)
        encoded_col_test = test_data[col].map(target_means_cats_adj)
        kfold = KFold(folds, shuffle=True, random_state=1).split(train_data[target_col].values)
        parts = []
        for (tr_in, val_ind) in kfold:
            (df_for_estimation, df_estimated) = (train_data.iloc[tr_in], train_data.iloc[val_ind])
            nrows_cat = df_for_estimation.groupby(col)[target_col].count()
            target_means_cats = df_for_estimation.groupby(col)[target_col].mean()
            target_means_cats_adj = (target_means_cats * nrows_cat + target_mean_global * alpha) / (nrows_cat + alpha)
            encoded_col_train_part = df_estimated[col].map(target_means_cats_adj)
            parts.append(encoded_col_train_part)
            encoded_col_train = pd.concat(parts, axis=0)
            encoded_col_train.fillna(target_mean_global, inplace=True)
        encoded_col = pd.concat([encoded_col_train, encoded_col_test], axis=0)
        encoded_col[encoded_col.isnull()] = target_mean_global
        encoded_cols.append(pd.DataFrame({'mean_' + target_col + '_' + col: encoded_col}))
    all_encoded = pd.concat(encoded_cols, axis=1)
    return (all_encoded.loc[train_data.index, :], all_encoded.loc[test_data.index, :])
train_mean_encoding = df_train[list(train_high_cat.columns)].copy()
train_mean_encoding['SalePrice'] = df_train['SalePrice']
target_col = 'SalePrice'
columns = train_mean_encoding.columns.tolist()
columns_test = columns
columns_test.remove('SalePrice')
test_mean_encoding = df_test[columns_test]
index_0 = list(range(0, 1459))
index_1 = list(range(1451, 2910))
test_mean_encoding = test_mean_encoding.rename(index=dict(zip(index_0, index_1)))
Mean_encoding = mean_encode(train_mean_encoding, test_mean_encoding, columns, target_col, alpha=5, folds=10)
train_high_cat_encoded = np.log(Mean_encoding[0].reset_index(drop=True))
test_high_cat_encoded = np.log(Mean_encoding[1].reset_index(drop=True))
from sklearn.preprocessing import OneHotEncoder
train_test_norm_cat = pd.concat([df_train[train_norm_cat], df_test[train_norm_cat]], axis=0, join='outer', ignore_index=True)
OHE = OneHotEncoder(sparse=False, handle_unknown='ignore')
train_test_norm_cat_OHE = pd.DataFrame(pd.DataFrame(OHE.fit_transform(train_test_norm_cat)))
train_test_norm_cat_OHE.columns = OHE.get_feature_names(train_test_norm_cat.columns.tolist())
NULLS = pd.DataFrame({'%_nulls': train_test_norm_cat_OHE.isin([0]).mean()})
NULLS = NULLS.reset_index().sort_values(ascending=False, by='%_nulls')
NULLS = NULLS.rename(columns={'index': 'Variable'})
DROP = NULLS.loc[(NULLS['%_nulls'] >= 0.99) | (NULLS['%_nulls'] <= 0.005), 'Variable'].values
train_test_norm_cat_OHE = train_test_norm_cat_OHE.drop(DROP, axis=1)
train_norm_cat_OHE = train_test_norm_cat_OHE.iloc[:1451,]
test_norm_cat_OHE = train_test_norm_cat_OHE.iloc[1451:,].reset_index(drop=True)
train_ordinal = pd.DataFrame()
test_ordinal = pd.DataFrame()
train_ordinal['OverallQual'] = df_train['OverallQual']
train_ordinal['OverallCond'] = df_train['OverallCond']
test_ordinal['OverallQual'] = df_test['OverallQual']
test_ordinal['OverallCond'] = df_test['OverallCond']
train_cont_balanced_default = df_train[train_cont_balanced].copy()
test_cont_balanced_default = df_test[train_cont_balanced].copy()
train_list = [train_high_cat_encoded, train_norm_cat_OHE, train_cont_balanced_default, train_ordinal]
X_train = pd.concat(train_list, axis=1)
y = np.log(df_train['SalePrice'])
test_list = [test_high_cat_encoded, test_norm_cat_OHE, test_cont_balanced_default, test_ordinal]
X_test = pd.concat(test_list, axis=1)
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold
CV = KFold(n_splits=10, random_state=999, shuffle=True)
CV_rep = RepeatedKFold(n_splits=10, n_repeats=3, random_state=999)
from sklearn import linear_model
import xgboost as xgb
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(X_train, y, test_size=0.1, random_state=999, shuffle=True)
import lightgbm as lgb
from scipy.stats import randint
from scipy.stats import uniform
from sklearn.preprocessing import RobustScaler
vars_for_scaling = train_high_cat_encoded.columns.tolist() + train_cont_balanced_default.columns.tolist()
Scaler = RobustScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
for column in vars_for_scaling:
    X_train_scaled[column] = Scaler.fit_transform(X_train[[column]])
    X_test_scaled[column] = Scaler.fit_transform(X_test[[column]])
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.pipeline import make_pipeline
base_learners = [('Lasso', linear_model.Lasso(tol=1e-07, alpha=0.00028, max_iter=3000)), ('El_Net', linear_model.ElasticNet(tol=1e-06, alpha=0.00044, l1_ratio=0.61, max_iter=4000)), ('XGB', xgb.XGBRegressor(use_label_encoder=False, eval_metric='rmse', n_estimators=5000, reg_alpha=0.1, reg_lambda=0.005, learning_rate=0.0125, max_depth=13, min_child_weight=4, gamma=0.04, subsample=0.7, colsample_bytree=0.6)), ('LGBM', lgb.LGBMRegressor(n_estimators=9000, reg_lambda=1.8, reg_alpha=0.01, min_child_samples=13, subsample=0.8, subsample_freq=11, num_leaves=101, max_depth=3, max_bin=160, learning_rate=0.005, colsample_bytree=0.1)), ('KNN', make_pipeline(RobustScaler(), KNeighborsRegressor(leaf_size=25, n_neighbors=9, p=1, weights='distance', metric='minkowski', algorithm='ball_tree'))), ('SVR', make_pipeline(RobustScaler(), SVR(kernel='rbf', C=10, epsilon=0.017, gamma=0.0007)))]
Final_stack = StackingRegressor(estimators=base_learners, final_estimator=linear_model.Lasso(tol=1e-07, alpha=0.00028, max_iter=3000), passthrough=True, verbose=False, cv=5)