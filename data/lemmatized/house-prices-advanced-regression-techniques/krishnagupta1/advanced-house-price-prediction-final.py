import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
print('train_dim:', _input1.shape)
_input1.head()
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print('test_dim:', _input0.shape)
_input0.head()
data = pd.concat([_input1.drop('SalePrice', axis=1), _input0], axis=0)
print(data.shape)
data.head()
Total = data.isna().sum()
Percentage = data.isna().mean()
nan = pd.concat([Total, Percentage], axis=1)
nan.columns = ['Total', 'Percentage']
nan[nan.Total > 0].sort_values('Total', ascending=False)
data = data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1, inplace=False)
print(data.shape)
data.head()
categorical = [features for features in data.columns if data[features].dtypes == 'O']
print('number of categorical features', len(categorical))
data[categorical].head()
categorical_Total = data[categorical].isna().sum()
categorical_percentage = data[categorical].isna().mean()
categorical_nan = pd.concat([categorical_Total, categorical_percentage], axis=1)
categorical_nan.columns = ['categorical_Total', 'categorical_percentage']
categorical_nan = categorical_nan[categorical_nan['categorical_Total'] > 0].sort_values('categorical_Total', ascending=False)
categorical_nan
col = categorical_nan.index
for column in col:
    mode = _input1[column].mode()[0]
    data[column] = data[column].fillna(mode)
data[categorical].isna().sum()[data[categorical].isna().sum() > 0]
data_temporal = [features for features in data.columns if 'Yr' in features or 'Year' in features]
print('number of temporal variables:', len(data_temporal))
data[data_temporal].head()
Temporal_Total = data[data_temporal].isna().sum()
Temporal_percentage = data[data_temporal].isna().mean()
Temporal_nan = pd.concat([Temporal_Total, Temporal_percentage], axis=1)
Temporal_nan.columns = ['Temporal_Total', 'Temporal_percentage']
Temporal_nan = Temporal_nan[Temporal_nan['Temporal_Total'] > 0].sort_values('Temporal_Total', ascending=False)
Temporal_nan
mode = data['GarageYrBlt'].mode()
data['GarageYrBlt'] = data['GarageYrBlt'].fillna(int(mode))
data[data_temporal].isna().sum()[data[data_temporal].isna().sum() > 0]
data_numerical = [features for features in data.columns if data[features].dtypes != 'O' and features not in data_temporal]
print('Number of Numerical Features:', len(data_numerical))
data[data_numerical].head()
Numerical_Total = data[data_numerical].isna().sum()
Numerical_percentage = data[data_numerical].isna().mean()
Numerical_nan = pd.concat([Numerical_Total, Numerical_percentage], axis=1)
Numerical_nan.columns = ['Numerical_Total', 'Numerical_percentage']
Numerical_nan = Numerical_nan[Numerical_nan['Numerical_Total'] > 0].sort_values('Numerical_Total', ascending=False)
Numerical_nan
col = Numerical_nan.index
for column in col:
    median = data[column].median()
    data[column] = data[column].fillna(median)
data[data_numerical].isna().sum()[data[data_numerical].isna().sum() > 0]
data = data.drop(['Street', 'Utilities', 'LandSlope', 'Condition2', 'RoofMatl', 'Heating', 'CentralAir', 'Functional', 'GarageQual'], axis=1)
data.shape
data[data_temporal].head()
data['RemodAddYear'] = data['YrSold'] - data['YearRemodAdd']
data['BuiltYear'] = data['YrSold'] - data['YearBuilt']
data['GarageBltYear'] = data['YrSold'] - data['GarageYrBlt']
data = data.drop(['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold'], axis=1, inplace=False)
data['RemodAddYear'][data['RemodAddYear'] < 0] = 0
data['BuiltYear'][data['BuiltYear'] < 0] = 0
data['GarageBltYear'][data['GarageBltYear'] < 0] = 0
column = [col for col in data.columns if data[col].dtypes != 'O']
for i in column:
    sns.boxplot(x=data[i])

def Outlires_NonG(df, col):
    IQR = df[col].quantile(0.75) - df[col].quantile(0.25)
    lower_bridge = int(df[col].quantile(0.25) - IQR * 1.5)
    upper_bridge = int(df[col].quantile(0.75) + IQR * 1.5)
    df.loc[df[col] > upper_bridge, col] = upper_bridge
    df.loc[df[col] < lower_bridge, col] = lower_bridge
for col in column:
    Outlires_NonG(data, col)
for i in column:
    sns.boxplot(x=data[i])
_input1['RemodAddYear'] = _input1['YrSold'] - _input1['YearRemodAdd']
_input1['BuiltYear'] = _input1['YrSold'] - _input1['YearBuilt']
_input1['GarageBltYear'] = _input1['YrSold'] - _input1['GarageYrBlt']
_input1.shape
abs(_input1.corr()['SalePrice']).sort_values(ascending=False)
abs(_input1.corr()['SalePrice']).sort_values(ascending=False).index
data1 = data.drop(['MSSubClass', 'OverallCond', 'MoSold', '3SsnPorch', 'LowQualFinSF', 'Id', 'MiscVal', 'BsmtHalfBath', 'BsmtFinSF2'], axis=1)
data1.shape
import scipy.stats as stat
import pylab
import seaborn as sns
import matplotlib.pyplot as plt

def plot_data(df, feature):
    con_col = []
    try:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        sns.distplot(df[feature])
        plt.subplot(1, 3, 2)
        df.boxplot(column=feature)
        plt.subplot(1, 3, 3)
        stat.probplot(df[feature], dist='norm', plot=pylab)
    except:
        con_col.append(col)
    return con_col

def nor(df1, col):
    df = df1.copy()
    con_col1 = []
    try:
        df[col + '_log'] = np.log(df[col] + 1)
        df[col + '_sqrt'] = df[col] ** (1 / 2)
        df[col + '_exp'] = df[col] ** (1 / 1.2)
        (df[col + '_boxcox'], parameters) = stat.boxcox(df[col] + 1)
        plt.figure(figsize=(18, 6))
        plt.subplot(1, 4, 1)
        stat.probplot(df[col + '_log'], dist='norm', plot=pylab)
        plt.subplot(1, 4, 2)
        stat.probplot(df[col + '_sqrt'], dist='norm', plot=pylab)
        plt.subplot(1, 4, 3)
        stat.probplot(df[col + '_exp'], dist='norm', plot=pylab)
        plt.subplot(1, 4, 4)
        stat.probplot(df[col + '_boxcox'], dist='norm', plot=pylab)
    except:
        con_col1.append(col)
    return con_col1
column1 = [col for col in data1.columns if data1[col].dtypes != 'O']
for col in column1:
    plot_data(data1, col)
    nor(data1, col)
    print('*' * 125)
boxcox = ['BsmtUnfSF', '1stFlrSF', 'GrLivArea']
exp = ['BsmtFinSF1', '2ndFlrSF', 'GarageBltYear']
sqrt = ['WoodDeckSF', 'OpenPorchSF', 'MasVnrArea', 'RemodAddYear']
for col in sqrt:
    data1[col] = data1[col] ** (1 / 2)
for col in exp:
    data1[col] = data1[col] ** (1 / 1.2)
for col in boxcox:
    (data1[col], parameters) = stat.boxcox(data1[col] + 1)

def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr
corr_features = correlation(data, 0.85)
print(len(set(corr_features)))
corr_features
data1 = data1.drop('GarageArea', axis=1)
data2 = pd.get_dummies(data1)
data2.shape
(_input1.shape, _input0.shape)
df_train = data2[:1460]
df_test = data2[1460:]
target = _input1['SalePrice']
(df_train.shape, df_test.shape, target.shape)
from sklearn.model_selection import train_test_split
(X_train, X_val, y_train, y_val) = train_test_split(df_train, target, test_size=0.3, random_state=42)
(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
import xgboost
xgb_final = xgboost.XGBRegressor(base_score=0.01, booster='gbtree', colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.5, gamma=0, gpu_id=-1, importance_type='gain', interaction_constraints='', learning_rate=0.025, max_delta_step=0, max_depth=5, min_child_weight=0, monotone_constraints='()', n_estimators=1250, n_jobs=12, num_parallel_tree=1, random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=0.7, tree_method='exact', validate_parameters=1, verbosity=None)