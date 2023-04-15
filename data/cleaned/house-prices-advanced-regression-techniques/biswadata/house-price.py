import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
df_train.head(4).style.background_gradient()
print(f'Train Data Shape: {df_train.shape}')
print(f'Test Data Shape: {df_test.shape}')
Sale_price = df_train.SalePrice
df_combined = pd.concat([df_train.drop('SalePrice', axis=1), df_test], axis=0)
df_combined.head(4).style.background_gradient()
df_combined.shape
replaced_data_values = dict({'Alley': 'No_alley_acces', 'BsmtQual': 'No_Basement', 'BsmtCond': 'No_Basement', 'BsmtExposure': 'No_Basement', 'BsmtFinType1': 'No_Basement', 'BsmtFinType2': 'No_Basement', 'FireplaceQu': 'No_Fireplace', 'GarageType': 'No_Garage', 'GarageFinish': 'No_Garage', 'GarageCond': 'No_Garage', 'GarageQual': 'No_Garage', 'PoolQC': 'No_Pool', 'MiscFeature': 'Nothing', 'Fence': 'No_Fence', 'MasVnrType': 'Nothing'})

def replace_data(df):
    for idx in replaced_data_values:
        df[idx].fillna(replaced_data_values[idx], inplace=True)
    return df
df_combined1 = replace_data(df_combined)
df_combined1.isnull().sum()
categorical = df_combined1.dtypes[df_combined.dtypes == 'object'].index
print(categorical)
numerical = df_combined1.dtypes[df_combined.dtypes != 'object'].index
print(numerical)
for i in categorical:
    df_combined1[i].fillna(df_combined1[i].mode(), inplace=True)
for i in numerical:
    df_combined1[i].fillna(df_combined1[i].median(), inplace=True)
dictn = {}
for i in range(len(df_combined1.isnull().sum())):
    if df_combined1.isnull().sum()[i] > 0:
        dictn[f'{df_combined1.columns[i]}'] = df_combined1.isnull().sum()[i]
dictn
df_combined1['MSZoning'].unique()
for i in categorical:
    df_combined1[i].replace(np.nan, 0, inplace=True)
dictn = {}
for i in range(len(df_combined1.isnull().sum())):
    if df_combined1.isnull().sum()[i] > 0:
        dictn[f'{df_combined1.columns[i]}'] = df_combined1.isnull().sum()[i]
dictn
df_combined1.head(2).style.background_gradient()
df_train.shape
df_combined2 = pd.get_dummies(df_combined1, drop_first=True)
df_combined2.head().style
df_train_processed = df_combined2[:1460]
df_train_processed.shape
df_train_processed['MSSubClass'].corr(Sale_price)
columns_to_maintain = []
for i in df_train_processed.columns:
    if abs(df_train_processed[i].corr(Sale_price)) > 0.4:
        columns_to_maintain.append(i)
columns_to_maintain
X = df_train_processed[columns_to_maintain]
y = Sale_price
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=12)
from sklearn.linear_model import LinearRegression
model1 = LinearRegression()