import warnings
warnings.filterwarnings('ignore')
import random
random.seed(1234)
import time
import re
from math import sqrt
from scipy import stats
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.impute import SimpleImputer as imp
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize': (16, 10)})
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import sklearn.metrics as skm
raw_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
raw_data.head()
raw_data.info()
raw_data.describe()
raw_data['SalePrice'].describe()
sns.histplot(data=raw_data, x='SalePrice')
plt.xlabel('Dollar Amount ($)')
plt.ylabel('Frequency (Count)')
plt.title('Distribution of House Sale Price')


def basic_processing(data, col_drop=[], col_rename={'SalePrice': 'target'}):
    """ Basic data processing: drop/rename columns, remove duplicate(s)
    
    Parameters
    ----------
    data : dataframe
        A dataset
    col_drop : list
        A list of column names to drop
    col_rename : dict
        A dictionary pairing the old and new column names desired

    Returns
    -------
    data
        a modified dataframe
    """
    if len(col_drop) > 0:
        data.drop(col_drop, axis=1, inplace=True)
    if col_rename:
        data.rename(columns=col_rename, inplace=True)
    data.drop_duplicates(keep='first', inplace=True)
    return data
df = basic_processing(raw_data)
df.head()

def identify_missing_val(data):
    """ Identify missing/na values
    
    Parameters
    ----------
    data : dataframe
        A dataset

    Returns
    -------
    data
        a dataframe with no missing values 
        either after imputation or original format
    """
    sum_na = data.isnull().sum().sum()
    print('%d null/na values found in the dataset.' % sum_na)
    if sum_na > 0:
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isna().transpose(), cmap='YlGnBu', cbar_kws={'label': 'Missing Data'})
        plt.xlabel('Features')
        plt.ylabel('Observations')

        null_cols = data.columns[data.isnull().any()].tolist()
        print('Those columns have missing values in those count and proportions:')
        for i in null_cols:
            col_null = data[i].isnull().sum()
            per_null = col_null / len(df[i])
            print('  - {}: {} ({:.2%})'.format(i, col_null, per_null))
    else:
        print('- No action needed')
        pass
    return data
df = identify_missing_val(df)
print("For all houses' LotFrontage, mean = {:.2f} and median = {:.2f}".format(df['LotFrontage'].mean(), df['LotFrontage'].median()))
print('For houses of shape: ')
for i in df['LotShape'].unique().tolist():
    df_i = df[df['LotShape'] == i]
    mean_frontage = df_i['LotFrontage'].mean()
    median_frontage = df_i['LotFrontage'].median()
    print(' -{}, mean LotFrontage = {:.2f} and median LotFrontage = {:.2f}'.format(i, mean_frontage, median_frontage))

def missing_value_fix(data, rep):
    data.dropna(subset=['Electrical'], inplace=True)
    data['LotFrontage'] = data.groupby('LotShape').LotFrontage.transform(lambda x: x.fillna(x.median()))
    for (i, j) in rep.items():
        data[j].fillna(i, inplace=True)
        if i in ['Unknown', 'No Basement', 'No Garage', 0]:
            for col in j:
                data[col].fillna(i, inplace=True)
    null_vals = data.isna().sum().sum()
    print('Afer imputation, we have missing {:d} values in our data.'.format(null_vals))
    return data
rep = {'No Alley': 'Alley', 'No Fireplace': 'FireplaceQu', 'No Pool': 'PoolQC', 'No Fence': 'Fence', 'No Misc': 'MiscFeature', 2010: 'GarageYrBlt', 'Unknown': ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'KitchenQual', 'Functional', 'SaleType'], 'No Basement': ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'], 'No Garage': ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'], 0: ['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea']}
df = missing_value_fix(df, rep)
df.head()

def outliers_detection(data, threshold=3):
    """ Function to detect outliers
    
    Parameters
    ----------
    data : dataframe
        A dataset
    threshold:
        The threshold at which a value is an outlier
        ±2.5σ from the µ

    Returns
    -------
    data
        a dataframe with no missing values 
        either after imputation or original format
    """
    print('At ±', threshold, 'standard deviation from the mean:')
    for col in data.columns[:-1]:
        if data[col].dtype.kind in 'iufc':
            var = data[col]
            z = np.abs(stats.zscore(var))
            pos = list(np.where(z > threshold)[0])
            no_of_outliers = len(pos)
            if no_of_outliers > 0:
                print('\t- ', col, 'variable contains', no_of_outliers, 'outliers')
        else:
            continue
outliers_detection(df)
sns.scatterplot(data=df, x='LotArea', y='target')
plt.xlabel('LotArea (in sq.)', fontsize=16)
plt.ylabel('Price', fontsize=16)
plt.title('House Size vs. Price', fontsize=20)

total = df['Neighborhood'].value_counts()[0]
per = df['Neighborhood'].value_counts(normalize=True)[0]
neigh_name = pd.DataFrame(df['Neighborhood'].value_counts()).index[0]
print('{} has the most houses sales with {} making up {:.2%} of all sales.'.format(neigh_name, total, per))
df_grouped = pd.DataFrame(df.groupby('Neighborhood')['target'].sum())
df_sorted = df_grouped.sort_values('target', ascending=False)
df_sorted['per_total'] = df_sorted['target'] / df_sorted['target'].sum()
neigh_name = df_sorted.index[0]
total = df_sorted['target'][0]
per = df_sorted['per_total'][0]
print('{} has the highest cumulative sales amount of ${:,} making up {:.2%} of all transactions.'.format(neigh_name, total, per))
total = df['MiscFeature'].value_counts()[1]
misc_name = pd.DataFrame(df['MiscFeature'].value_counts()).index[1]
print('For houses with miscellaneous features, {} is the most prevalent in {} houses.'.format(misc_name, total))
misc = df[df['MiscFeature'] == 'Shed']['MiscVal']
sale = df[df['MiscFeature'] == 'Shed']['target']
avg_value_added = np.average(misc)
per_sale = np.average(misc / sale)
print('{} brings ${:.2f} of value added making {:.2%} of the house sale price on average.'.format(misc_name, avg_value_added, per_sale))

def datetime_encoding(df, cols, mapping):
    """ Creating time intervals
    """
    for c in cols:
        df[c] = df[c].apply(lambda x: 2010 if x > 2010 else x)
        df[c] = pd.cut(df[df[c] != 0][c], bins=6, precision=0).astype(str)
        df[c].fillna(0, inplace=True)
    enc_ord = ce.OrdinalEncoder(mapping=mapping, return_df=True)
    df_final = enc_ord.fit_transform(df)
    return df_final
date_cols_mapping = [{'col': 'YrSold', 'mapping': {2006: 0, 2007: 1, 2008: 2, 2009: 3, 2010: 4}}, {'col': 'MoSold', 'mapping': {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 12: 11}}, {'col': 'YearRemodAdd', 'mapping': {'(1950.0, 1960.0]': 0, '(1960.0, 1970.0]': 1, '(1970.0, 1980.0]': 2, '(1980.0, 1990.0]': 3, '(1990.0, 2000.0]': 4, '(2000.0, 2010.0]': 5}}, {'col': 'YearBuilt', 'mapping': {'(1872.0, 1895.0]': 0, '(1895.0, 1918.0]': 1, '(1918.0, 1941.0]': 2, '(1941.0, 1964.0]': 3, '(1964.0, 1987.0]': 4, '(1987.0, 2010.0]': 5}}, {'col': 'GarageYrBlt', 'mapping': {0: 0, '(1900.0, 1918.0]': 1, '(1918.0, 1937.0]': 2, '(1937.0, 1955.0]': 3, '(1955.0, 1973.0]': 4, '(1973.0, 1992.0]': 5, '(1992.0, 2010.0]': 6}}]
cols = ['YearRemodAdd', 'YearBuilt', 'GarageYrBlt']
df = datetime_encoding(df, cols, date_cols_mapping)
check_cols = ['YrSold', 'MoSold', 'YearRemodAdd', 'YearBuilt', 'GarageYrBlt']
df[check_cols].head()

def categorical_encoding(df, binary_vars, nominal_vars, ordinal_cols_mapping):
    """ Function encoding  categorical variables
    
    Parameters
    ----------
    data : dataframe
        A dataset
        
    binary_vars, nominal_vars:
        List of binary and nominal categorical variables, respectively
    
    ordinal_cols_mapping:
        List of dictionary mapping the corresponding order of each category

    Returns
    -------
    data_encoded
        a dataframe with all categorical encoding transfomation
    """
    binenc = ce.BinaryEncoder(cols=binary_vars, return_df=True)
    df = binenc.fit_transform(df)
    for c in nominal_vars:
        df[c] = df[c].astype('category')
        df[c] = df[c].cat.codes
    ordenc = ce.OrdinalEncoder(mapping=ordinal_cols_mapping, return_df=True)
    df_final = ordenc.fit_transform(df)
    return df_final
bins = ['CentralAir']
noms = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'Functional', 'GarageType', 'MiscFeature', 'SaleType', 'SaleCondition', 'Electrical']
cat_ord_mapping = [{'col': 'LotShape', 'mapping': {'Reg': 0, 'IR1': 1, 'IR2': 2, 'IR3': 3}}, {'col': 'LandContour', 'mapping': {'Low': 0, 'Lvl': 1, 'Bnk': 2, 'HLS': 3}}, {'col': 'Utilities', 'mapping': {'ELO': 0, 'NoSeWa': 1, 'NoSewr': 2, 'AllPub': 3}}, {'col': 'LandSlope', 'mapping': {'Gtl': 0, 'Mod': 1, 'Sev': 2}}, {'col': 'OverallQual', 'mapping': {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9}}, {'col': 'OverallCond', 'mapping': {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9}}, {'col': 'ExterQual', 'mapping': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}}, {'col': 'ExterCond', 'mapping': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}}, {'col': 'BsmtQual', 'mapping': {'No Basement': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}}, {'col': 'BsmtCond', 'mapping': {'No Basement': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}}, {'col': 'BsmtExposure', 'mapping': {'No Basement': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}}, {'col': 'BsmtFinType1', 'mapping': {'No Basement': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}}, {'col': 'BsmtFinType2', 'mapping': {'No Basement': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}}, {'col': 'HeatingQC', 'mapping': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}}, {'col': 'KitchenQual', 'mapping': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}}, {'col': 'FireplaceQu', 'mapping': {'No Fireplace': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}}, {'col': 'GarageFinish', 'mapping': {'No Garage': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}}, {'col': 'GarageQual', 'mapping': {'No Garage': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}}, {'col': 'GarageCond', 'mapping': {'No Garage': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}}, {'col': 'PavedDrive', 'mapping': {'N': 0, 'P': 1, 'Y': 2}}, {'col': 'PoolQC', 'mapping': {'No Pool': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}}, {'col': 'Fence', 'mapping': {'No Fence': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}}]
df = categorical_encoding(df, binary_vars=bins, nominal_vars=noms, ordinal_cols_mapping=cat_ord_mapping)
df.head()

def feature_scaling(data, cont_vars):
    """ Function standardizing numerical variable of different scales
    
    Parameters
    ----------
    data : dataframe
        A dataset
        
    Returns
    -------
    data
        A standardized dataframe
    """
    scaler = MinMaxScaler()
    for col in cont_vars:
        data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))
    return data
num_vars = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
df = feature_scaling(df, num_vars)
df[num_vars].head()
matrix_corr = df.corr()
matrix_corr = np.round(matrix_corr.unstack(), 2)
strong_rel = matrix_corr[(abs(matrix_corr) >= 0.75) & (abs(matrix_corr) != 1.0)]
strong_rel
matrix_corr = df.corr()
matrix_sale = matrix_corr.filter(regex='^Sale', axis=1)
matrix_sale = np.round(matrix_sale.unstack(), 2)
strong_rel_sale = matrix_sale[(abs(matrix_sale) >= 0.75) & (abs(matrix_sale) != 1.0)]
strong_rel_sale

def model_comparison(df, models, metric='neg_mean_squared_log_error'):
    """ Function to split data into training and testing set
    
    Parameters
    ----------
    data : dataframe
        A dataset
    models: A dictionary
        A pre-defined dictionary with each model name and applicable function
    
    optional train_test params
    
    Returns
    -------
    models_df
        A dataframe with each model performance
    """
    X = df.drop(columns=['target'], axis=1)
    y = df['target']
    models_perf = {'Models': [], 'CV_RMSLE_mean': [], 'CV_RMSLE_std': []}
    for model in models:
        cv_results = cross_validate(model, X, y, cv=5, scoring=metric)
        cv_rmsle = np.sqrt(abs(cv_results['test_score']))
        m = str(model)
        models_perf['Models'].append(m[:m.find('(')])
        models_perf['CV_RMSLE_mean'].append(np.mean(cv_rmsle))
        models_perf['CV_RMSLE_std'].append(np.std(cv_rmsle))
    models_df = pd.DataFrame(models_perf, columns=['Models', 'CV_RMSLE_mean', 'CV_RMSLE_std'])
    return models_df
models_simple = [DecisionTreeRegressor(), SGDRegressor(), SVR()]
model_comparison(df, models_simple)
models_ensemble = [RandomForestRegressor(), BaggingRegressor(), GradientBoostingRegressor(), XGBRegressor()]
model_comparison(df, models_ensemble)

def hyperparameter_tuning(data, model, search_space, metric='neg_mean_squared_log_error'):
    """ Conduct hyperparameter tuning using GridSearch
    
    Parameters
    ----------
    data : dataframe
        Dataset
    model: Model function
        Corresponding function in Sklearn
    search_space: A dictionary
        Possible values for each parameter to iterate
    metric: String
        Evaluation metric name per Scikit-learn documentation
    
    Returns
    -------
    models_df
        A
    """
    X = df.drop(columns=['target'], axis=1)
    y = df['target']
    search = GridSearchCV(estimator=model, param_grid=search_space, cv=5, scoring=metric, refit=metric)