import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
pd.set_option('display.max_columns', None)
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1
_input1.columns
_input1.dtypes
_input1.describe()
missing_val = [features for features in _input1.columns if _input1[features].isnull().sum() > 1]
for feature in missing_val:
    print(feature, np.round(_input1[feature].isnull().mean(), 3), '% missing values')
sns.heatmap(_input1.isnull(), yticklabels=False, cbar=False, cmap='viridis')
for i in _input1.columns:
    if _input1[i].dtypes == 'object':
        _input1[i] = _input1[i].fillna(_input1[i].mode()[0], inplace=False)
    else:
        _input1[i] = _input1[i].fillna(_input1[i].median(), inplace=False)
print(_input1)
sns.heatmap(_input1.isnull(), yticklabels=False, cbar=False, cmap='viridis')
_input1.head()
sns.lineplot(x='YrSold', y='SalePrice', data=_input1)
plt.title('Year build vs Sale Price')
plt.xlabel('Year Sold')
plt.ylabel('Price')
sns.lineplot(x='YearBuilt', y='SalePrice', data=_input1)
plt.title('Built Year vs Sale Price')
plt.xlabel('Year Build')
plt.ylabel('Price')
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant')), ('encoder', OneHotEncoder(handle_unknown='ignore'))])
numerical_val = [features for features in _input1.columns if _input1[features].dtypes != 'O']
categorical_val = [features for features in _input1.columns if _input1[features].dtypes == 'O']
preprocessor = ColumnTransformer(transformers=[('numeric', numeric_transformer, numerical_val), ('categorical', categorical_transformer, categorical_val)])
_input1.head()
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input0
_input0.columns
_input0.dtypes
_input0.describe()
missing_val = [features for features in _input0.columns if _input0[features].isnull().sum() > 1]
for feature in missing_val:
    print(feature, np.round(_input0[feature].isnull().mean(), 3), '% missing values')
sns.heatmap(_input0.isnull(), yticklabels=False, cbar=False, cmap='viridis')
for i in _input0.columns:
    if _input0[i].dtypes == 'object':
        _input0[i] = _input0[i].fillna(_input0[i].mode()[0], inplace=False)
    else:
        _input0[i] = _input0[i].fillna(_input0[i].median(), inplace=False)
print(_input0)
sns.heatmap(_input0.isnull(), yticklabels=False, cbar=False, cmap='viridis')
sns.barplot(y='YrSold', x='SaleType', data=_input0)
plt.title('Year Sold vs Sale Type')
plt.xlabel('Sale Type')
plt.ylabel('Year Sold')
sns.barplot(y='YearBuilt', x='SaleType', data=_input0)
plt.title('Year Built vs Sale Type')
plt.xlabel('Sale Type')
plt.ylabel('Year Built')
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant')), ('encoder', OneHotEncoder(handle_unknown='ignore'))])
numerical_val = [features for features in _input0.columns if _input0[features].dtypes != 'O']
categorical_val = [features for features in _input0.columns if _input0[features].dtypes == 'O']
preprocessor = ColumnTransformer(transformers=[('numeric', numeric_transformer, numerical_val), ('categorical', categorical_transformer, categorical_val)])
from sklearn.model_selection import train_test_split
X = _input1.iloc[:, :-1]
y = _input1.iloc[:, -1]
(X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=101)
X_test
from xgboost import XGBRegressor
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', XGBRegressor())])