import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
pd.set_option('display.max_columns', None)
house_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
house_train
house_train.columns
house_train.dtypes
house_train.describe()
missing_val = [features for features in house_train.columns if house_train[features].isnull().sum() > 1]
for feature in missing_val:
    print(feature, np.round(house_train[feature].isnull().mean(), 3), '% missing values')
sns.heatmap(house_train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
for i in house_train.columns:
    if house_train[i].dtypes == 'object':
        house_train[i].fillna(house_train[i].mode()[0], inplace=True)
    else:
        house_train[i].fillna(house_train[i].median(), inplace=True)
print(house_train)
sns.heatmap(house_train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
house_train.head()
sns.lineplot(x='YrSold', y='SalePrice', data=house_train)
plt.title('Year build vs Sale Price')
plt.xlabel('Year Sold')
plt.ylabel('Price')
sns.lineplot(x='YearBuilt', y='SalePrice', data=house_train)
plt.title('Built Year vs Sale Price')
plt.xlabel('Year Build')
plt.ylabel('Price')
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant')), ('encoder', OneHotEncoder(handle_unknown='ignore'))])
numerical_val = [features for features in house_train.columns if house_train[features].dtypes != 'O']
categorical_val = [features for features in house_train.columns if house_train[features].dtypes == 'O']
preprocessor = ColumnTransformer(transformers=[('numeric', numeric_transformer, numerical_val), ('categorical', categorical_transformer, categorical_val)])
house_train.head()
house_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
house_test
house_test.columns
house_test.dtypes
house_test.describe()
missing_val = [features for features in house_test.columns if house_test[features].isnull().sum() > 1]
for feature in missing_val:
    print(feature, np.round(house_test[feature].isnull().mean(), 3), '% missing values')
sns.heatmap(house_test.isnull(), yticklabels=False, cbar=False, cmap='viridis')
for i in house_test.columns:
    if house_test[i].dtypes == 'object':
        house_test[i].fillna(house_test[i].mode()[0], inplace=True)
    else:
        house_test[i].fillna(house_test[i].median(), inplace=True)
print(house_test)
sns.heatmap(house_test.isnull(), yticklabels=False, cbar=False, cmap='viridis')
sns.barplot(y='YrSold', x='SaleType', data=house_test)
plt.title('Year Sold vs Sale Type')
plt.xlabel('Sale Type')
plt.ylabel('Year Sold')
sns.barplot(y='YearBuilt', x='SaleType', data=house_test)
plt.title('Year Built vs Sale Type')
plt.xlabel('Sale Type')
plt.ylabel('Year Built')
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant')), ('encoder', OneHotEncoder(handle_unknown='ignore'))])
numerical_val = [features for features in house_test.columns if house_test[features].dtypes != 'O']
categorical_val = [features for features in house_test.columns if house_test[features].dtypes == 'O']
preprocessor = ColumnTransformer(transformers=[('numeric', numeric_transformer, numerical_val), ('categorical', categorical_transformer, categorical_val)])
from sklearn.model_selection import train_test_split
X = house_train.iloc[:, :-1]
y = house_train.iloc[:, -1]
(X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=101)
X_test
from xgboost import XGBRegressor
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', XGBRegressor())])