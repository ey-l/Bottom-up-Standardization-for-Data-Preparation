import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head(n=5)
_input0.tail(n=5)
(_input1['DataType'], _input0['DataType']) = ('training', 'testing')
_input0.insert(_input0.shape[1] - 1, 'SalePrice', np.nan)
df_data = pd.concat([_input1, _input0], ignore_index=True)

def boxplot(categorical_x: list or str, numerical_y: list or str, data: pd.DataFrame, figsize: tuple=(4, 3), ncols: int=5, nrows: int=None) -> plt.figure:
    """ Return a box plot applied for categorical variable in x-axis vs numerical variable in y-axis.
    
    Args:
        categorical_x (list or str): The categorical variable in x-axis.
        numerical_y (list or str): The numerical variable in y-axis.
        data (pd.DataFrame): The data to plot.
        figsize (tuple): The matplotlib figure size width and height in inches. Default to (4, 3).
        ncols (int): The number of columns for axis in the figure. Default to 5.
        nrows (int): The number of rows for axis in the figure. Default to None.
    
    Returns:
        plt.figure: The plot figure.
    """
    (categorical_x, numerical_y) = ([categorical_x] if type(categorical_x) == str else categorical_x, [numerical_y] if type(numerical_y) == str else numerical_y)
    if nrows is None:
        nrows = (len(categorical_x) * len(numerical_y) - 1) // ncols + 1
    (fig, axes) = plt.subplots(figsize=(figsize[0] * ncols, figsize[1] * nrows), ncols=ncols, nrows=nrows)
    axes = axes.flatten()
    _ = [sns.boxplot(x=vj, y=vi, data=data, ax=axes[i * len(categorical_x) + j]) for (i, vi) in enumerate(numerical_y) for (j, vj) in enumerate(categorical_x)]
    return fig

def scatterplot(numerical_x: list or str, numerical_y: list or str, data: pd.DataFrame, figsize: tuple=(4, 3), ncols: int=5, nrows: int=None) -> plt.figure:
    """ Return a scatter plot applied for numerical variable in x-axis vs numerical variable in y-axis.
    
    Args:
        numerical_x (list or str): The numerical variable in x-axis.
        numerical_y (list or str): The numerical variable in y-axis.
        data (pd.DataFrame): The data to plot.
        figsize (tuple): The matplotlib figure size width and height in inches. Default to (4, 3).
        ncols (int): The number of columns for axis in the figure. Default to 5.
        nrows (int): The number of rows for axis in the figure. Default to None.
    
    Returns:
        plt.figure: The plot figure.
    """
    (numerical_x, numerical_y) = ([numerical_x] if type(numerical_x) == str else numerical_x, [numerical_y] if type(numerical_y) == str else numerical_y)
    if nrows is None:
        nrows = (len(numerical_x) * len(numerical_y) - 1) // ncols + 1
    (fig, axes) = plt.subplots(figsize=(figsize[0] * ncols, figsize[1] * nrows), ncols=ncols, nrows=nrows)
    axes = axes.flatten()
    _ = [sns.scatterplot(x=vj, y=vi, data=data, ax=axes[i * len(numerical_x) + j], rasterized=True) for (i, vi) in enumerate(numerical_y) for (j, vj) in enumerate(numerical_x)]
    return fig
df_data.describe(include='all')
col_convert = ['MSSubClass']
df_data[col_convert] = df_data[col_convert].astype('object')
col_number = df_data.select_dtypes(include=['number']).columns.tolist()
print('features type number:\n items %s\n length %d' % (col_number, len(col_number)))
col_object = df_data.select_dtypes(include=['object']).columns.tolist()
print('features type object:\n items %s\n length %d' % (col_object, len(col_object)))
_ = df_data.hist(bins=20, figsize=(20, 15))
df_data['SalePrice'] = np.log1p(df_data['SalePrice'])
df_data['MiscVal'] = np.log1p(df_data['MiscVal'])
col_number = df_data.select_dtypes(include=['number']).columns.drop(['Id']).tolist()
col_object = df_data.select_dtypes(include=['object']).columns.tolist()
_ = scatterplot(col_number, 'SalePrice', df_data[df_data['DataType'] == 'training'])
_ = boxplot(col_object, 'SalePrice', df_data[df_data['DataType'] == 'training'])
col_number = df_data.select_dtypes(include=['number']).columns.drop(['Id']).tolist()
col_object = df_data.select_dtypes(include=['object']).columns.tolist()
_ = scatterplot(col_number, 'LotFrontage', df_data)
_ = boxplot(col_object, 'LotFrontage', df_data)
df_data['LotFrontage'] = df_data['LotFrontage'].fillna(df_data.groupby(['Neighborhood'])['LotFrontage'].transform('mean'))
df_data.loc[(df_data['LotFrontage'] > 200) & (df_data['DataType'] == 'trainnig'), 'DataType'] = 'excluded'
df_data.loc[(df_data['LotArea'] > 100000) & (df_data['DataType'] == 'training'), 'DataType'] = 'excluded'
df_data.loc[(df_data['BsmtFinSF1'] > 4000) & (df_data['DataType'] == 'training'), 'DataType'] = 'excluded'
df_data.loc[(df_data['TotalBsmtSF'] > 5000) & (df_data['DataType'] == 'training'), 'DataType'] = 'excluded'
df_data.loc[(df_data['1stFlrSF'] > 4000) & (df_data['DataType'] == 'training'), 'DataType'] = 'excluded'
df_data.loc[(df_data['GrLivArea'] > 4500) & (df_data['DataType'] == 'training'), 'DataType'] = 'excluded'
df_data.loc[(df_data['OpenPorchSF'] > 500) & (df_data['SalePrice'] < 11) & (df_data['DataType'] == 'training'), 'DataType'] = 'excluded'
col_convert = ['LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea']
df_data[col_convert] = df_data[col_convert].fillna(0)
df_data['TotalSF'] = df_data['TotalBsmtSF'] + df_data['GrLivArea']
df_data['TotalPorch'] = df_data['OpenPorchSF'] + df_data['EnclosedPorch'] + df_data['3SsnPorch'] + df_data['ScreenPorch']
df_data['TotalArea'] = df_data['TotalSF'] + df_data['TotalPorch'] + df_data['GarageArea'] + df_data['WoodDeckSF']
col_convert = ['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd']
df_data[col_convert] = df_data[col_convert].fillna(0)
df_data['TotalBathBsmt'] = df_data['BsmtFullBath'] + 0.5 * df_data['BsmtHalfBath']
df_data['TotalBathAbvGrd'] = df_data['FullBath'] + 0.5 * df_data['HalfBath']
df_data['TotalRmsAbvGrdIncBath'] = df_data['TotRmsAbvGrd'] + df_data['TotalBathAbvGrd']
df_data['TotalRms'] = df_data['TotalRmsAbvGrdIncBath'] + df_data['TotalBathBsmt']
df_data['AreaPerRmsBsmt'] = df_data['TotalBsmtSF'] / (df_data['TotalBathBsmt'] + 1)
df_data['AreaPerRmsGrLivAbvGrd'] = df_data['GrLivArea'] / (df_data['TotalRmsAbvGrdIncBath'] + 1)
df_data['AreaPerRmsTotal'] = df_data['TotalSF'] / (df_data['TotalRms'] + 1)
col_convert = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
df_data[col_convert] = df_data[col_convert].replace('Ex', 5).replace('Gd', 4).replace('TA', 3).replace('Fa', 2).replace('Po', 1).replace('NA', 0)
df_data[col_convert] = df_data[col_convert].fillna(0).astype(int)
df_data['ExterQualCond'] = df_data['ExterQual'] * df_data['ExterCond']
df_data['BsmtQualCond'] = df_data['BsmtQual'] * df_data['BsmtCond']
df_data['GarageQualCond'] = df_data['GarageQual'] * df_data['GarageCond']
df_data['OverallQualCond'] = df_data['OverallQual'] * df_data['OverallCond']
col_convert = ['BsmtExposure']
df_data[col_convert] = df_data[col_convert].replace('Gd', 4).replace('Av', 3).replace('Mn', 2).replace('No', 1).replace('NA', 0)
df_data[col_convert] = df_data[col_convert].fillna(0).astype(int)
col_convert = ['BsmtFinType1', 'BsmtFinType2']
df_data[col_convert] = df_data[col_convert].replace('GLQ', 6).replace('ALQ', 5).replace('BLQ', 4).replace('Rec', 3).replace('LwQ', 2).replace('Unf', 1).replace('NA', 0)
df_data[col_convert] = df_data[col_convert].fillna(0).astype(int)
col_convert = ['GarageFinish']
df_data[col_convert] = df_data[col_convert].replace('Fin', 3).replace('RFn', 2).replace('Unf', 1).replace('NA', 0)
df_data[col_convert] = df_data[col_convert].fillna(0).astype(int)
col_convert = ['Fence']
df_data[col_convert] = df_data[col_convert].replace('GdPrv', 4).replace('MnPrv', 3).replace('GdWo', 2).replace('MnWw', 1).replace('NA', 0)
df_data[col_convert] = df_data[col_convert].fillna(0).astype(int)
df_data['GarageYrBlt'] = df_data['GarageYrBlt'].fillna(df_data['YearBuilt'])
df_data['YearBuiltRemod'] = df_data['YearRemodAdd'] - df_data['YearBuilt']
df_data['YearBuiltSold'] = df_data['YrSold'] - df_data['YearBuilt']
df_data['YearRemodSold'] = df_data['YrSold'] - df_data['YearRemodAdd']
df_data['YearGarageSold'] = df_data['YrSold'] - df_data['GarageYrBlt']
df_season = df_data.loc[df_data['DataType'] == 'training'].groupby(['YrSold', 'MoSold'], as_index=False).agg({'SalePrice': 'mean'})
(fig, axes) = plt.subplots(figsize=(20, 3))
_ = sns.pointplot(x='MoSold', y='SalePrice', data=df_season, join=True, hue='YrSold')
df_data['Utilities'] = df_data['Utilities'].fillna('ELO')
df_data['SaleType'] = df_data['SaleType'].fillna('Oth')
col_fillnas = ['MSZoning', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Electrical', 'Functional']
for col_fillna in col_fillnas:
    df_data[col_fillna] = df_data[col_fillna].fillna(df_data[col_fillna].value_counts().idxmax())
col_fillnas = ['Alley', 'GarageType']
for col_fillna in col_fillnas:
    df_data[col_fillna] = df_data[col_fillna].fillna('NA')
col_fillnas = ['GarageCars', 'MiscFeature']
df_data[col_fillnas] = df_data[col_fillnas].fillna(0)
if df_data.isna().any().any():
    print(df_data.loc[:, df_data.columns[df_data.isna().any()].tolist()].describe(include='all'))
else:
    print('no null entry')
col_number = df_data.select_dtypes(include=['number']).columns.drop(['MiscVal', 'SalePrice', 'YearBuiltRemod', 'YearBuiltSold', 'YearRemodSold', 'YearGarageSold']).tolist()
for col_transform in col_number:
    skewness = scipy.stats.skew(df_data[col_transform].dropna())
    if skewness > 0.75:
        df_data[col_transform] = np.log1p(df_data[col_transform])
col_number = df_data.select_dtypes(include=['number']).columns.drop(['Id']).tolist()
col_object = df_data.select_dtypes(include=['object']).columns.tolist()
_ = scatterplot(col_number, 'SalePrice', df_data[df_data['DataType'] == 'training'])
_ = boxplot(col_object, 'SalePrice', df_data[df_data['DataType'] == 'training'])
df_data['SalePrice'] = df_data['SalePrice'].fillna(0)
df_data = pd.get_dummies(df_data, columns=None, drop_first=True)
df_data.describe(include='all')
df_data.info()
corr = df_data[df_data['DataType_training'] == 1].corr()
(fig, axes) = plt.subplots(figsize=(200, 150))
heatmap = sns.heatmap(corr, annot=True, cmap=plt.cm.RdBu, fmt='.1f', square=True, vmin=-0.8, vmax=0.8)
x = df_data[df_data['DataType_training'] == 1].drop(['Id', 'SalePrice', 'DataType_training', 'DataType_testing'], axis=1)
y = df_data.loc[df_data['DataType_training'] == 1, 'SalePrice']