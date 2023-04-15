import sklearn
import seaborn as sns
import matplotlib.mlab as mlab
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
import xgboost
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
path1 = '_data/input/house-prices-advanced-regression-techniques/train.csv'
path2 = '_data/input/house-prices-advanced-regression-techniques/test.csv'
df_train = pd.read_csv(path1)
df_train.head()
df_test = pd.read_csv(path2)
df_test.head()
(df_train.shape, df_test.shape)
df = df_train.append(df_test).reset_index(drop=True)
df.shape
df.value_counts(np.where(df['SalePrice'] > 0, '1', '0'))

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    grab_col_names for given dataframe

    :param dataframe:
    :param cat_th:
    :param car_th:
    :return:
    """
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == 'O']
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != 'O']
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == 'O']
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != 'O']
    num_cols = [col for col in num_cols if col not in num_but_cat]
    print(f'Observations: {dataframe.shape[0]}')
    print(f'Variables: {dataframe.shape[1]}')
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return (cat_cols, cat_but_car, num_cols, num_but_cat)
(cat_cols, cat_but_car, num_cols, num_but_cat) = grab_col_names(df)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(), 'Ratio': 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)

for col in cat_cols:
    cat_summary(df, col)
df[num_cols].describe([0.05, 0.1, 0.25, 0.5, 0.75, 0.95, 0.99]).T

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)

    print('#####################################')
for col in num_cols:
    num_summary(df, col)
df['SalePrice'].describe([0.05, 0.1, 0.25, 0.5, 0.75, 0.8, 0.9, 0.95, 0.99]).T
sns.set(rc={'figure.figsize': (6, 6)})
df['SalePrice'].hist(bins=100)

df = df.loc[~(df.SalePrice > 600000),]
df['SalePrice'].hist(bins=100)

print('Skew: %f' % df['SalePrice'].skew())
np.log1p(df['SalePrice']).hist(bins=50)

print('Skew: %f' % np.log1p(df['SalePrice']).skew())
df.head()
corr = df[num_cols].corr()
sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap='YlGnBu')


def high_correlated_cols(dataframe, plot=False, corr_th=0.7):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap='RdBu')

    return drop_list
high_correlated_cols(df, plot=False)

def outlier_thresholds(dataframe, variable, low_quantile=0.1, up_quantile=0.9):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return (low_limit, up_limit)

def check_outlier(dataframe, col_name):
    (low_limit, up_limit) = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    (low_limit, up_limit) = outlier_thresholds(dataframe, variable)
    dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit
for col in num_cols:
    if col != 'SalePrice':
        print(col, check_outlier(df, col))
for col in num_cols:
    if col != 'SalePrice':
        replace_with_thresholds(df, col)

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end='\n')
    if na_name:
        return na_columns
missing_values_table(df)
no_cols = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']
for col in no_cols:
    df[col].fillna('No', inplace=True)
missing_values_table(df)
df.shape

def quick_missing_imp(data, num_method='median', cat_length=20, target='SalePrice'):
    variables_with_na = [col for col in data.columns if data[col].isnull().sum() > 0]
    temp_target = data[target]
    print('# BEFORE')
    print(data[variables_with_na].isnull().sum(), '\n\n')
    data = data.apply(lambda x: x.fillna(x.mode()[0]) if x.dtype == 'O' and len(x.unique()) <= cat_length else x, axis=0)
    if num_method == 'mean':
        data = data.apply(lambda x: x.fillna(x.mean()) if x.dtype != 'O' else x, axis=0)
    elif num_method == 'median':
        data = data.apply(lambda x: x.fillna(x.median()) if x.dtype != 'O' else x, axis=0)
    data[target] = temp_target
    print("# AFTER \n Imputation method is 'MODE' for categorical variables!")
    print(" Imputation method is '" + num_method.upper() + "' for numeric variables! \n")
    print(data[variables_with_na].isnull().sum(), '\n\n')
    return data
df = quick_missing_imp(df, num_method='median', cat_length=17)

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ':', len(dataframe[col].value_counts()))
        print(pd.DataFrame({'COUNT': dataframe[col].value_counts(), 'RATIO': dataframe[col].value_counts() / len(dataframe), 'TARGET_MEAN': dataframe.groupby(col)[target].mean()}), end='\n\n\n')
rare_analyser(df, 'SalePrice', cat_cols)
df['MSZoning'].value_counts()
df.loc[df['MSZoning'] == 'RH', 'MSZoning'] = 'RM'
df.loc[df['MSZoning'] == 'FV', 'MSZoning'] = 'FV + C (all)'
df.loc[df['MSZoning'] == 'C (all)', 'MSZoning'] = 'FV + C (all)'
df['MSZoning'].value_counts()
sns.set(rc={'figure.figsize': (5, 5)})
bins = 50
plt.hist(df['LotArea'], bins, alpha=0.5, density=True)

print(df['LotArea'].max())
print(df['LotArea'].mean())
New_LotArea = pd.Series(['Studio', 'Small', 'Middle', 'Large', 'Dublex', 'Luxury'], dtype='category')
df['New_LotArea'] = New_LotArea
df.loc[df['LotArea'] <= 2000, 'New_LotArea'] = New_LotArea[0]
df.loc[(df['LotArea'] > 2000) & (df['LotArea'] <= 4000), 'New_LotArea'] = New_LotArea[1]
df.loc[(df['LotArea'] > 4000) & (df['LotArea'] <= 6000), 'New_LotArea'] = New_LotArea[2]
df.loc[(df['LotArea'] > 6000) & (df['LotArea'] <= 8000), 'New_LotArea'] = New_LotArea[3]
df.loc[(df['LotArea'] > 10000) & (df['LotArea'] <= 12000), 'New_LotArea'] = New_LotArea[4]
df.loc[df['LotArea'] > 12000, 'New_LotArea'] = New_LotArea[5]
df['New_LotArea'].value_counts()
df['LotShape'].value_counts()
df.loc[df['LotShape'] == 'IR1', 'LotShape'] = 'IR'
df.loc[df['LotShape'] == 'IR2', 'LotShape'] = 'IR'
df.loc[df['LotShape'] == 'IR3', 'LotShape'] = 'IR'
df['LotShape'].value_counts()
df['ExterCond'].value_counts()
df['ExterCond'] = np.where(df.ExterCond.isin(['Fa', 'Po']), 'FaPo', df['ExterCond'])
df['ExterCond'] = np.where(df.ExterCond.isin(['Ex', 'Gd']), 'ExGd', df['ExterCond'])
df['ExterCond'].value_counts()
df['GarageQual'].value_counts()
df['GarageQual'] = np.where(df.GarageQual.isin(['Fa', 'Po']), 'FaPo', df['GarageQual'])
df['GarageQual'] = np.where(df.GarageQual.isin(['Ex', 'Gd']), 'ExGd', df['GarageQual'])
df['GarageQual'] = np.where(df.GarageQual.isin(['ExGd', 'TA']), 'ExGd', df['GarageQual'])
df['GarageQual'].value_counts()
df['BsmtFinType1'].value_counts()
df['BsmtFinType2'].value_counts()
df['BsmtFinType1'] = np.where(df.BsmtFinType1.isin(['GLQ', 'ALQ']), 'RareExcellent', df['BsmtFinType1'])
df['BsmtFinType1'] = np.where(df.BsmtFinType1.isin(['BLQ', 'LwQ', 'Rec']), 'RareGood', df['BsmtFinType1'])
df['BsmtFinType2'] = np.where(df.BsmtFinType2.isin(['GLQ', 'ALQ']), 'RareExcellent', df['BsmtFinType2'])
df['BsmtFinType2'] = np.where(df.BsmtFinType2.isin(['BLQ', 'LwQ', 'Rec']), 'RareGood', df['BsmtFinType2'])
df['BsmtFinType1'].value_counts()
df['BsmtFinType2'].value_counts()
df['Condition1'].value_counts()
df.loc[(df['Condition1'] == 'Feedr') | (df['Condition1'] == 'Artery') | (df['Condition1'] == 'RRAn') | (df['Condition1'] == 'PosA') | (df['Condition1'] == 'RRAe'), 'Condition1'] = 'AdjacentCondition'
df.loc[(df['Condition1'] == 'RRNn') | (df['Condition1'] == 'PosN') | (df['Condition1'] == 'RRNe'), 'Condition1'] = 'WithinCondition'
df.loc[df['Condition1'] == 'Norm', 'Condition1'] = 'NormalCondition'
df['Condition1'].value_counts()
df['Condition2'].value_counts()
df.drop('Condition2', axis=1, inplace=True)
df['BldgType'].value_counts()
df['BldgType'] = np.where(df.BldgType.isin(['1Fam', '2fmCon']), 'Normal', df['BldgType'])
df['BldgType'] = np.where(df.BldgType.isin(['TwnhsE', 'Twnhs', 'Duplex']), 'Big', df['BldgType'])
df['BldgType'].value_counts()
df['TotalQual'] = df[['OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageQual', 'GarageCond', 'Fence']].sum(axis=1)
df['Overall'] = df[['OverallQual', 'OverallCond']].sum(axis=1)
df['NEW_TotalFlrSF'] = df['1stFlrSF'] + df['2ndFlrSF']
df['NEW_TotalBsmtFin'] = df.BsmtFinSF1 + df.BsmtFinSF2
df['NEW_PorchArea'] = df.OpenPorchSF + df.EnclosedPorch + df.ScreenPorch + df['3SsnPorch'] + df.WoodDeckSF
df['NEW_TotalHouseArea'] = df.NEW_TotalFlrSF + df.TotalBsmtSF
df['NEW_TotalSqFeet'] = df.GrLivArea + df.TotalBsmtSF
df['NEW_TotalFullBath'] = df.BsmtFullBath + df.FullBath
df['NEW_TotalHalfBath'] = df.BsmtHalfBath + df.HalfBath
df['NEW_TotalBath'] = df['NEW_TotalFullBath'] + df['NEW_TotalHalfBath'] * 0.5
df['NEW_LotRatio'] = df.GrLivArea / df.LotArea
df['NEW_RatioArea'] = df.NEW_TotalHouseArea / df.LotArea
df['NEW_GarageLotRatio'] = df.GarageArea / df.LotArea
df['NEW_Restoration'] = df.YearRemodAdd - df.YearBuilt
df['NEW_HouseAge'] = df.YrSold - df.YearBuilt
df['NEW_RestorationAge'] = df.YrSold - df.YearRemodAdd
df['NEW_GarageAge'] = df.GarageYrBlt - df.YearBuilt
df['NEW_GarageRestorationAge'] = np.abs(df.GarageYrBlt - df.YearRemodAdd)
df['NEW_GarageSold'] = df.YrSold - df.GarageYrBlt
df.head()
drop_list = ['Street', 'Alley', 'LandContour', 'Utilities', 'LandSlope', 'Heating', 'PoolQC', 'MiscFeature', 'Neighborhood', 'KitchenAbvGr', 'CentralAir', 'Functional']
df.drop(drop_list, axis=1, inplace=True)
(cat_cols, cat_but_car, num_cols, num_but_cat) = grab_col_names(df)

def label_encoder(dataframe, binary_col):
    labelencoder = preprocessing.LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe
binary_cols = [col for col in df.columns if df[col].dtypes == 'O' and len(df[col].unique()) == 2]
for col in binary_cols:
    label_encoder(df, col)
df.head()

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
df = one_hot_encoder(df, cat_cols, drop_first=True)
df.head()
missing_values_table(df)
train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()].drop('SalePrice', axis=1)
y = np.log1p(df[df['SalePrice'].notnull()]['SalePrice'])
X = train_df.drop(['Id', 'SalePrice'], axis=1)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=17)
models = [('LR', LinearRegression()), ('Ridge', Ridge()), ('Lasso', Lasso()), ('ElasticNet', ElasticNet()), ('KNN', KNeighborsRegressor()), ('CART', DecisionTreeRegressor()), ('RF', RandomForestRegressor()), ('SVR', SVR()), ('GBM', GradientBoostingRegressor()), ('XGBoost', XGBRegressor(objective='reg:squarederror')), ('LightGBM', LGBMRegressor())]
for (name, regressor) in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring='neg_mean_squared_error')))
    print(f'RMSE: {round(rmse, 4)} ({name}) ')
gboost_model = GradientBoostingRegressor(loss='squared_error')
rmse = np.mean(np.sqrt(-cross_val_score(gboost_model, X, y, cv=5, scoring='neg_mean_squared_error')))
gboost_params = {'learning_rate': [0.1, 0.01, 0.03], 'max_depth': [5, 6, 8], 'n_estimators': [100, 200, 300], 'subsample': [0.5, 0.8, 1]}