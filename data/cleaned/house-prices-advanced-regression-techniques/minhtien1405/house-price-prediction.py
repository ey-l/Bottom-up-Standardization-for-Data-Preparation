import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from ipywidgets import widgets
import plotly.express as px
train_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train_df.head()
train_df.describe()
clean_df = train_df.copy()
clean_df.drop(columns=['Street', 'LandContour', 'Utilities', 'LandSlope', 'Condition1', 'Condition2', 'BldgType', 'RoofMatl', 'ExterCond', 'BsmtCond', 'BsmtFinType2', 'Heating', 'CentralAir', 'Electrical', 'LowQualFinSF', 'BsmtHalfBath', 'KitchenAbvGr', 'Functional', 'GarageQual', 'GarageCond', 'PavedDrive', '3SsnPorch', 'PoolArea', 'MiscVal', 'SaleType', 'SaleCondition', 'Alley', 'PoolQC', 'Fence', 'MiscFeature'], inplace=True)
clean_df.head()
discret_columns = ['MasVnrType', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1', 'FireplaceQu', 'GarageType', 'GarageFinish']
for col in discret_columns:
    most_frequent_value = clean_df[col].value_counts().index[0]
    clean_df[col].fillna(most_frequent_value, inplace=True)
continuous_columns = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
for col in continuous_columns:
    most_frequent_value = clean_df[col].value_counts().index[0]
    clean_df[col].fillna(clean_df[col].mean(), inplace=True)

def get_list_of_low_corr_columns(df):
    corr_df = df.corr()
    columns = corr_df[abs(corr_df['SalePrice']) < 0.1]['SalePrice'].index
    columns = [c for c in columns if c in df.columns]
    return list(columns)
clean_df.drop(columns=get_list_of_low_corr_columns(clean_df), inplace=True)
clean_df.head()
normalizing_columns = ['LotFrontage', 'LotArea', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'BsmtFullBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'ScreenPorch']
for col in normalizing_columns:
    clean_df[col] = (clean_df[col] - clean_df[col].mean()) / clean_df[col].std()
column_informations = {}
num_values = len(train_df)
for col in train_df.columns:
    num_unique = train_df[col].nunique()
    num_nulls = round(train_df[col].isna().sum() / num_values, 2)
    d_type = train_df.dtypes[col]
    if num_unique < 30:
        info_str = '['
        value_counts = train_df[col].value_counts()
        single_value_weight = round(value_counts.iloc[0] / num_values, 2)
        for (index, value) in value_counts.items():
            info_str += f'{value} X {index}, '
        column_informations[col] = {'d_type': d_type, 'discret': True, 'percentage_of_missing_values': num_nulls, 'single_value_weight': single_value_weight, 'min': 0.0, 'max': 0.0, 'mean': 0.0, 'median': 0.0, 'info_str': info_str[:-2] + ']'}
    elif d_type == 'int64' or d_type == 'float64':
        column_informations[col] = {'d_type': d_type, 'discret': False, 'percentage_of_missing_values': num_nulls, 'single_value_weight': 0.0, 'min': train_df[col].min(), 'max': train_df[col].max(), 'mean': round(train_df[col].mean(), 2), 'median': round(train_df[col].median(), 2), 'info_str': ''}
    else:
        column_informations[col] = {'d_type': d_type, 'discret': False, 'percentage_of_missing_values': num_nulls, 'min': '-', 'max': '-', 'mean': '-', 'median': '-', 'info_str': ''}
info_df = pd.DataFrame.from_dict(column_informations, orient='index')

print(len(info_df[info_df['discret'] == True]))

def get_list_of_numeric_columns(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    return [c for c in df.select_dtypes(include=numerics).columns if c in df.columns]

def get_list_of_non_numeric_cat_cols(df, info_df=info_df):
    list_of_categorical_cols = list(info_df.loc[info_df['discret']].index)
    list_of_categorical_cols = [c for c in list_of_categorical_cols if c in df.columns]
    list_of_numerical_cols = get_list_of_numeric_columns(df)
    return [c for c in list_of_categorical_cols if c not in list_of_numerical_cols]
categorical_columns = get_list_of_non_numeric_cat_cols(clean_df)
for col in categorical_columns:
    clean_df[col] = clean_df[col].astype('category').cat.codes
clean_df.head()
y = clean_df['SalePrice']
clean_df.drop(columns=['SalePrice'], inplace=True)
import xgboost as xgb
import warnings
from xgboost import plot_tree
warnings.filterwarnings(action='ignore', category=UserWarning)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
(X_train, X_test, y_train, y_test) = train_test_split(clean_df, y)