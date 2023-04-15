import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
submission = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
train.head()
print('Number of rows in Training dataset: ', train.shape[0])
print('Number of rows in Testing dataset: ', test.shape[0])
print(f'Number of columns({train.shape[1] - 1} features + 1 Traget) in Train: ', train.shape[1])
print(f'Number of columns({test.shape[1]} features) in Test: ', test.shape[1])
print('Columns: ', train.columns)
train.info()
train.describe().T

import matplotlib.pyplot as plt
import seaborn as sns
sns.scatterplot(x='LotArea', y='SalePrice', data=train)
for col in train.columns:
    plt.figure(figsize=(12, 6))
    sns.histplot(train[col])
for col in train.columns:
    if train[col].dtype != 'object':
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x=col, y='SalePrice', data=train)
corr_matrix = train.corr()
corr_matrix['SalePrice'].sort_values(ascending=False)
from pandas.plotting import scatter_matrix
features_matrix = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF']
scatter_matrix(train[features_matrix], figsize=(24, 12))
train_features = train.drop('SalePrice', axis=1)
train_labels = train['SalePrice']
(train_features.shape, train_labels.shape)
numeric = train_features._get_numeric_data().columns
categoric = [i for i in train_features.columns if i not in numeric]
train_features.isnull().sum().index
null_columns = []
for i in range(len(train_features.isnull().sum().index)):
    if train_features.isnull().sum()[i] != 0:
        print(train_features.isnull().sum().index[i], '-', train_features.isnull().sum()[i])
        null_columns.append(train_features.isnull().sum().index[i])
null_num_columns = [i for i in null_columns if i in numeric]
null_cat_columns = [i for i in null_columns if i not in numeric]
null_cat_columns
train_num = train_features[numeric]
train_cat = train_features[categoric]
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('std_scalar', StandardScaler())])
from sklearn.compose import ColumnTransformer
final_pipeline = ColumnTransformer([('num', num_pipeline, list(train_num.columns)), ('cat', OneHotEncoder(handle_unknown='ignore'), list(train_cat.columns))])
train_final = final_pipeline.fit_transform(train_features)
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()