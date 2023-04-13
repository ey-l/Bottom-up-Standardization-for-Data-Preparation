import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import CategoricalDtype
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import mutual_info_regression
from scipy import stats
from math import ceil
plt.rcParams.update({'font.size': 12, 'xtick.labelsize': 15, 'ytick.labelsize': 15, 'axes.labelsize': 15, 'axes.titlesize': 20})
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
_input1 = _input1.dropna(axis=0, subset=['SalePrice'])
X = _input1.drop(['SalePrice'], axis=1)
y = _input1.SalePrice
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 30)
info = pd.DataFrame(X.dtypes, columns=['Dtype'])
info['Unique'] = X.nunique().values
info['Null'] = X.isnull().sum().values
info
X.dtypes.value_counts()
y.describe()
correlation_matrix = _input1.corr()
mask = np.triu(correlation_matrix.corr())
sns.set(font_scale=1.1)
plt.figure(figsize=(20, 20), dpi=140)
sns.heatmap(correlation_matrix, annot=True, fmt='.1f', cmap='coolwarm', square=True, mask=mask, linewidths=1, cbar=False)
print(len(X), len(_input0))

def show_null_values(X, X_test):
    null_values_train = X.isnull().sum()
    null_values_test = _input0.isnull().sum()
    null_values = pd.DataFrame(null_values_train)
    null_values['Test Data'] = null_values_test.values
    null_values = null_values.rename(columns={0: 'Train Data'}, inplace=False)
    null_values = null_values.loc[(null_values['Train Data'] != 0) | (null_values['Test Data'] != 0)]
    null_values = null_values.sort_values(by=['Train Data', 'Test Data'], ascending=False)
    print('Total miising values:', null_values.sum(), sep='\n')
    return null_values
show_null_values(X, _input0)
null_cols = [col for col in X.columns if X[col].isnull().sum() > len(X) / 2]
null_cols
X = X.drop(null_cols, axis=1, inplace=False)
_input0 = _input0.drop(null_cols, axis=1, inplace=False)
print('Total missing values:')
print('Training data\t', X.isnull().sum().sum())
print('Testing data\t', _input0.isnull().sum().sum())
object_cols = X.select_dtypes('object').columns
len(object_cols)
(fig, ax) = plt.subplots(nrows=ceil(len(object_cols) / 4), ncols=4, figsize=(22, 1.4 * len(object_cols)), sharey=True, dpi=120)
for (col, subplot) in zip(object_cols, ax.flatten()):
    freq = X[col].value_counts()
    subplot.ticklabel_format(style='plain')
    plt.ylim([0, 800000])
    plt.subplots_adjust(wspace=0.1, hspace=0.4)
    for tick in subplot.get_xticklabels():
        tick.set_rotation(45)
    sns.violinplot(data=X, x=col, y=y, order=freq.index, ax=subplot)
X.Utilities.value_counts()
_input0.Utilities.value_counts()
X = X.drop('Utilities', axis=1, inplace=False)
_input0 = _input0.drop('Utilities', axis=1, inplace=False)
df = pd.concat([X, _input0])
df1 = pd.DataFrame()
df1['Age'] = df['YrSold'] - df['YearBuilt']
df1['AgeRemodel'] = df['YrSold'] - df['YearRemodAdd']
Years = ['YrSold', 'YearBuilt', 'YearRemodAdd']
year_cols = ['YrSold', 'YearBuilt', 'AgeRemodel', 'Age']
df_1 = pd.concat([df, df1], axis=1).loc[:, year_cols]
X_1 = df_1.loc[X.index, :]
X_1.sample()
sns.set(style='whitegrid')
sns.set(rc={'figure.figsize': (11.7, 8.27), 'font.size': 20, 'axes.titlesize': 20, 'axes.labelsize': 20}, style='white')
(fig, ax) = plt.subplots(1, 4, figsize=(20, 6), dpi=100)
for (col, i) in zip(year_cols, [0, 1, 2, 3]):
    sns.scatterplot(x=X_1.loc[:, col], y=y, ax=ax[i], hue=X.ExterQual, palette='pastel')
fig.tight_layout()
fig.text(0.5, 1, 'Distribution of SalesPrice with respect to years columns', size=20, ha='center', va='center')
X_1.corrwith(y)
df2 = pd.DataFrame()
df2['Remodel'] = df['YearRemodAdd'] != df['YearBuilt']
df2['Garage'] = df['GarageQual'].notnull()
df2['Fireplace'] = df['FireplaceQu'].notnull()
df2['Bsmt'] = df['BsmtQual'].notnull()
df2['Masonry'] = df['MasVnrType'].notnull()
df2 = df2.replace([False, True], [0, 1])
df2.sample()
object_cols = df.select_dtypes(include=['object']).columns
df[object_cols].nunique().sort_values()
ordinal_cols = [i for i in object_cols if 'QC' in i or 'Qu' in i or 'Fin' in i or ('Cond' in i and 'Condition' not in i)]
df.loc[:, ordinal_cols] = df.loc[:, ordinal_cols].fillna('NA')
print('Column Names: [Unique Categories in each column]')
{col: [*df[col].unique()] for col in ordinal_cols}
ordinal_cols1 = [i for i in object_cols if 'QC' in i or 'Qu' in i or ('Cond' in i and 'Condition' not in i)]
df.loc[:, ordinal_cols1] = df.loc[:, ordinal_cols1].replace(['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], [0, 1, 2, 3, 4, 5])
ordinal_cols2 = ['BsmtFinType1', 'BsmtFinType2']
df.loc[:, ordinal_cols2] = df.loc[:, ordinal_cols2].replace(['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'], [0, 1, 2, 3, 4, 5, 6])
ordinal_cols3 = ['BsmtExposure']
df.loc[:, ordinal_cols3] = df.loc[:, ordinal_cols3].fillna('NA')
df.loc[:, ordinal_cols3] = df.loc[:, ordinal_cols3].replace(['NA', 'No', 'Mn', 'Av', 'Gd'], [0, 1, 2, 3, 4])
ordinal_cols4 = ['LotShape']
df.loc[:, ordinal_cols4] = df.loc[:, ordinal_cols4].replace(['Reg', 'IR1', 'IR2', 'IR3'], [0, 1, 2, 3])
ordinal_cols5 = ['GarageFinish']
df.loc[:, ordinal_cols5] = df.loc[:, ordinal_cols5].replace(['NA', 'Unf', 'RFn', 'Fin'], [0, 1, 2, 3])
ordinal_cols6 = ['Functional']
df.loc[:, ordinal_cols3] = df.loc[:, ordinal_cols3].fillna('Mod')
df.loc[:, ordinal_cols6] = df.loc[:, ordinal_cols6].replace(['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'], list(range(8)))
o_columns = ordinal_cols1 + ordinal_cols2 + ordinal_cols3 + ordinal_cols4 + ordinal_cols5 + ordinal_cols6
df.loc[:, o_columns].dtypes.value_counts()
Bath_cols = [i for i in df.columns if 'Bath' in i]
Bath_cols
SF_cols = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']
df[SF_cols + Bath_cols] = df[SF_cols + Bath_cols].fillna(0)
df3 = pd.DataFrame()
df3['Liv_Qual'] = (df.OverallQual + df.OverallCond / 3) * df.GrLivArea
df3['GarageArea_Qual'] = (df.GarageQual + df.GarageCond / 3) * df.GarageArea * df.GarageCars
df3['BsmtArea_Qual'] = df.BsmtQual * df.BsmtCond / 3 * df.TotalBsmtSF
df3['LivLotRatio'] = df.GrLivArea / df.LotArea
df3['Spaciousness'] = (df['1stFlrSF'] + df['2ndFlrSF']) / df.TotRmsAbvGrd
df3['TotalSF'] = df[SF_cols].sum(axis=1)
df3['TotalBath'] = df.FullBath + df.BsmtFullBath + df.HalfBath / 2 + df.BsmtHalfBath / 2
df3.sample()
df4 = pd.DataFrame()
Porches = ['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
df4['PorchTypes'] = df[Porches].gt(0.0).sum(axis=1)
df4.sample()
df5 = pd.DataFrame()
df5['MedNhbdArea'] = df.groupby('Neighborhood')['GrLivArea'].transform('median')
df5.sample()
df6 = pd.DataFrame()
df6 = pd.get_dummies(df.BldgType, prefix='Bldg')
df6 = df6.mul(df.GrLivArea, axis=0)
before = df.shape[1]
cat_columns_to_drop = [cname for cname in X.select_dtypes(['object', 'category', 'bool']).columns if X[cname].nunique() > 10]
df = df.drop(cat_columns_to_drop, axis=1, inplace=False)
after = df.shape[1]
print(f'Number of columns reduced from {before} to {after}')
cat_columns = list(df.select_dtypes('object').columns)
before = df[cat_columns].nunique().sum()
df.HouseStyle.value_counts(normalize=True)
for col in cat_columns:
    df[col] = df[col].mask(df[col].map(df[col].value_counts(normalize=True)) < 0.01, 'Other')
after = df[cat_columns].nunique().sum()
print(f'Number of unique categories reduced from {before} to {after}')
df.HouseStyle.value_counts(normalize=True) * 100
df.dtypes.value_counts()
features_nom = ['MSSubClass'] + cat_columns
for name in features_nom:
    df[name] = df[name].astype('category')
    if 'NA' not in df[name].cat.categories:
        df[name] = df[name].cat.add_categories('NA')
for colname in df.select_dtypes(['category']):
    df[colname] = df[colname].cat.codes
df.dtypes.value_counts()
df.shape
df = df.drop(Years + Porches, axis=1, inplace=False)
df = pd.concat([df, df1, df2, df3, df4, df5, df6], axis=1)
df.sample()
df.shape
X = df.loc[X.index, :]
_input0 = df.loc[_input0.index, :]
print(X.shape, _input0.shape, sep='\n')
my_imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
imputed_X = pd.DataFrame(my_imputer.fit_transform(X))
imputed_X_test = pd.DataFrame(my_imputer.transform(_input0))
imputed_X.columns = X.columns
imputed_X_test.columns = _input0.columns
imputed_X.index = X.index
imputed_X_test.index = _input0.index
X = imputed_X
_input0 = imputed_X_test
show_null_values(X, _input0)
X_y = X.copy()
X_y['SalesPrice'] = y
X_y.sample()

def univariate_numerical_plot(df, x):
    (fig, ax) = plt.subplots(1, 3, figsize=(20, 6), dpi=100)
    sns.histplot(data=df, x=x, kde=True, ax=ax[0], bins=min(df[x].nunique(), 10), kde_kws={'bw_adjust': 3})
    sns.despine(bottom=True, left=True)
    ax[0].set_title('histogram')
    ax[0].set_xlabel(xlabel=x)
    sns.boxplot(data=df, x=x, ax=ax[1])
    ax[1].set_title('boxplot')
    ax[1].set_ylabel(ylabel=x)
    sns.scatterplot(x=df[x], y=y, ax=ax[2], hue=y, palette='coolwarm')
    plt.legend([], [], frameon=False)
    fig.subplots_adjust(top=0.85, bottom=0.15, left=0.2, hspace=0.8)
    fig.patch.set_linewidth(10)
    fig.patch.set_edgecolor('cornflowerblue')
    fig.tight_layout()
    fig.text(0.5, 1, f'Distribution of {x}', size=25, ha='center', va='center')
univariate_numerical_plot(X_y, 'SalesPrice')

def make_mi_scores(X, y):
    X = X.copy()
    mi_scores = mutual_info_regression(X.select_dtypes('number'), y, random_state=0)
    mi_scores = pd.DataFrame(mi_scores.round(2), columns=['MI_Scores'], index=X.select_dtypes('number').columns)
    return mi_scores
mi_scores = make_mi_scores(X, y)
linear_corr = pd.DataFrame(X.corrwith(y).round(2), columns=['Lin_Correlation'])
corr_with_price = pd.concat([mi_scores, linear_corr], axis=1)
corr_with_price = corr_with_price.sort_values('MI_Scores', ascending=False)
corr_with_price
top_features = corr_with_price.index[1:6]
for feature in top_features:
    univariate_numerical_plot(X, feature)
before = X.shape[1]
X.dtypes.value_counts()
threshold = 0.01
numerical_cols = [cname for cname in X.select_dtypes('number').columns if corr_with_price.MI_Scores[cname] > threshold]
selected_cols = numerical_cols
X = X[selected_cols]
_input0 = _input0[selected_cols]
after = X.shape[1]
print(f'Out of {before} features, {after} fetures are having MI_Scores more than {threshold}.')
info = pd.DataFrame(X.dtypes, columns=['Dtype'])
info['Unique'] = X.nunique().values
info['Null'] = X.isnull().sum().values
info.sort_values(['Dtype', 'Unique'])
xgb = XGBRegressor(eval_metric='rmse')
param_grid = [{'subsample': [0.5], 'n_estimators': [1400], 'max_depth': [5], 'learning_rate': [0.02], 'colsample_bytree': [0.4], 'colsample_bylevel': [0.5], 'reg_alpha': [1], 'reg_lambda': [1], 'min_child_weight': [2]}]
grid_search = GridSearchCV(xgb, param_grid, cv=3, verbose=1, scoring='neg_root_mean_squared_error')