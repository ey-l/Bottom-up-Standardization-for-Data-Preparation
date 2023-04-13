import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1
_input1.hist(bins=50, figsize=(20, 20))
_input1.SalePrice.hist()
_input1.info()
_input1.describe()
from sklearn.model_selection import train_test_split
(train_set, test_set) = train_test_split(_input1, test_size=0.2, random_state=10)
len(test_set)
len(train_set)
train = train_set
corr_matrix = train.corr()
corr_matrix.SalePrice.sort_values(ascending=False)
corr_matrix.SalePrice.sort_values(ascending=False)[1:11]
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
housing = train.drop('SalePrice', axis=1)
target = train.SalePrice
target
housing.isna().sum().sort_values(ascending=False)
housing_incomplete_rows = housing[housing.isna().any(axis=1)]
housing_incomplete_rows.head()
housing_incomplete_rows.isna().sum().sort_values(ascending=False)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
temp = []
for i in housing.columns:
    if is_numeric_dtype(train[i]):
        temp.append(i)
housing_num = housing[temp]
temp2 = []
for i in housing.columns:
    if is_string_dtype(housing[i]):
        temp2.append(i)
housing_string = housing[temp2]
housing_string.shape
housing_num.shape
housing.shape
housing_cat = housing_string.apply(lambda x: x.astype('category'))
housing_num.hist(bins=50, figsize=(20, 20))
housing_num.corrwith(target, axis=0).sort_values(ascending=False)
num_drop_cols = ['BsmtFinSF2', 'BsmtUnfSF', 'KitchenAbvGr', 'YearBuilt', 'YrSold', 'LowQualFinSF', 'MoSold', 'BsmtUnfSF', 'GarageCars']
cat_drop_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
housing_num.isna().sum().sort_values(ascending=False) / len(housing_num) * 100
housing_num.LotFrontage.describe()
housing_cat.isna().sum().sort_values(ascending=False)[:15] / len(housing_cat) * 100
housing_num = housing_num.drop(num_drop_cols, axis=1, inplace=False)
housing_cat = housing_cat.drop(cat_drop_cols, axis=1, inplace=False)
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder(handle_unknown='ignore')
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('std_scaler', StandardScaler())])
cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('cat', cat_encoder)])
from sklearn.compose import ColumnTransformer
num_attribs = housing_num.columns
cat_attribs = housing_cat.columns
full_pipeline = ColumnTransformer([('num', num_pipeline, num_attribs), ('cat', cat_pipeline, cat_attribs)])
housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()