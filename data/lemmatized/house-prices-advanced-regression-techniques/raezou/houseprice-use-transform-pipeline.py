import numpy as np
import pandas as pd
import sklearn
import matplotlib as mpl
import matplotlib.pyplot as plt
sklearn.__version__
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1.head()
_input1 = _input1.drop('Id', axis=1)
_input1.head()
from sklearn.model_selection import train_test_split
(train_data, test_data) = train_test_split(_input1, test_size=0.2, random_state=42)
train_data.info()
train_data.shape
housing = train_data.copy()
corr_matrix = housing.corr()
corr_matrix['SalePrice'].sort_values(ascending=False)
from pandas.plotting import scatter_matrix
attributes = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF']
scatter_matrix(housing[attributes], figsize=(12, 12))
X_train = train_data.drop('SalePrice', axis=1)
y_train = train_data['SalePrice'].copy()
X_train.head()
y_train.head()
X_train.dtypes
X_train.shape
X_train['OverallQual'].value_counts()
X_train['ExterQual'].value_counts()
X_train['BsmtFinType1'].value_counts()
X_train['BsmtFinType2'].value_counts()
X_train['HeatingQC'].value_counts()
X_train['LowQualFinSF'].value_counts()
X_train['KitchenQual'].value_counts()
X_train['FireplaceQu'].value_counts()
X_train['PoolQC'].value_counts()
X_train['Fence'].value_counts()
X_train['GarageQual'].value_counts()

def getOrdinalPip(order):
    return Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OrdinalEncoder(categories=order, handle_unknown='use_encoded_value', unknown_value=-1)), ('scaler', StandardScaler())])
ordinal_columns = ['HeatingQC', 'GarageQual', 'FireplaceQu', 'KitchenQual', 'ExterQual']

def drop_ordinal(df):
    X_train_dump = df.drop(columns=ordinal_columns)
    return X_train_dump
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer

def get_categorical_columns(df):
    categorical_columns_selector = selector(dtype_include=object)
    categorical_columns = categorical_columns_selector(drop_ordinal(df))
    return categorical_columns

def get_numerical_columns(df):
    numerical_columns_selector = selector(dtype_exclude=object)
    numerical_columns = numerical_columns_selector(df)
    return numerical_columns
get_numerical_columns(X_train)
get_categorical_columns(X_train)

def get_ordinal_pipeline(order):
    return Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OrdinalEncoder(categories=order, handle_unknown='error', unknown_value=None)), ('scaler', StandardScaler())])
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def my_transformation(df):
    df = df.copy()
    numerical_columns = get_numerical_columns(df)
    nominal_columns = get_categorical_columns(df)
    ordinal_columns = ['GarageQual']
    ordinal_columns1 = ['FireplaceQu']
    ordinal_columns2 = ['HeatingQC']
    order1 = [['Po', 'Fa', 'TA', 'Gd', 'Ex']]
    ordinal_columns3 = ['KitchenQual']
    ordinal_columns4 = ['ExterQual']
    order2 = [['Fa', 'TA', 'Gd', 'Ex']]
    numerical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
    nominal_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))])
    ordinal_pipeline1 = get_ordinal_pipeline(order1)
    ordinal_pipeline2 = get_ordinal_pipeline(order2)
    preprocessor = ColumnTransformer([('numerical_transformer', numerical_pipeline, numerical_columns), ('nominal_transformer', nominal_pipeline, nominal_columns), ('ordinal_transformer', ordinal_pipeline1, ordinal_columns), ('ordinal_transformer1', ordinal_pipeline1, ordinal_columns1), ('ordinal_transformer2', ordinal_pipeline1, ordinal_columns2), ('ordinal_transformer3', ordinal_pipeline2, ordinal_columns3), ('ordinal_transformer4', ordinal_pipeline2, ordinal_columns4)])