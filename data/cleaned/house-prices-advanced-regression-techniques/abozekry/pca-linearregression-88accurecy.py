import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import TruncatedSVD
df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df.head()
df.drop(['Id', 'Alley', 'MasVnrArea', 'GarageYrBlt', 'PoolQC', 'Fence', 'MiscFeature', 'GarageYrBlt', 'FireplaceQu', 'YearBuilt', 'YearRemodAdd', 'YrSold', 'BsmtFinSF2', '2ndFlrSF', 'LowQualFinSF', 'WoodDeckSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal'], axis=1, inplace=True)
x_train = df.drop(columns='SalePrice')
y_train = df['SalePrice']
x_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
numeric_columns = x_train.select_dtypes(exclude='object').columns
numeric_columns
cat_columns = x_train.select_dtypes(include='object').columns
cat_columns
numeric_feature = Pipeline(steps=[('handlingmissing', SimpleImputer(strategy='median')), ('scaling', StandardScaler(with_mean=False))])
numeric_feature
cat_feature = Pipeline(steps=[('handlingmissing', SimpleImputer(strategy='most_frequent')), ('encoding', OneHotEncoder()), ('scaling', StandardScaler(with_mean=False))])
cat_feature
processing = ColumnTransformer([('numeic', numeric_feature, numeric_columns), ('cat', cat_feature, cat_columns)])
processing
final_pipe = Pipeline(steps=[('processing', processing), ('pca', TruncatedSVD(n_components=210, random_state=0)), ('modeling', LinearRegression())])
final_pipe