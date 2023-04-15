import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import seaborn as sns
import matplotlib.pyplot as plt
dataset = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
dataset.shape
dataset.columns
dataset.dtypes.unique()
dataset.head()
len(dataset.select_dtypes(include=['O']).columns)
len(dataset.select_dtypes(include=['int64']).columns)
len(dataset.select_dtypes(include=['float64']).columns)
saleprice_corr = dataset.corr()['SalePrice']
saleprice_corr
X = dataset[['Neighborhood', 'OverallQual', 'YearBuilt', 'ExterCond', 'TotalBsmtSF', 'GrLivArea', 'SalePrice']]
plt.scatter(X['GrLivArea'], X['SalePrice'])
plt.scatter(X['TotalBsmtSF'], X['SalePrice'])
plt.scatter(X['OverallQual'], X['SalePrice'])
plt.scatter(X['YearBuilt'], X['SalePrice'])
sns.pairplot(X, hue='ExterCond')
sns.pairplot(X, hue='Neighborhood')
sns.boxplot(x=X['OverallQual'], y=X['SalePrice'], palette='rainbow')
sns.boxplot(x=X['ExterCond'], y=X['SalePrice'], palette='rainbow')
plt.figure(figsize=(20, 10))
sns.boxplot(y=X['Neighborhood'], x=X['SalePrice'], palette='rainbow')
plt.figure(figsize=(20, 10))
sns.boxplot(x=X['YearBuilt'], y=X['SalePrice'], palette='rainbow')
X = dataset[['FullBath', 'OverallQual', 'YearBuilt', 'TotalBsmtSF', 'GrLivArea', 'GarageCars', 'SalePrice']]
plt.scatter(X['FullBath'], X['SalePrice'])
plt.scatter(X['GarageCars'], X['SalePrice'])
sns.pairplot(X)
total_missing_values_X = X.isnull().sum().sort_values(ascending=False)
total_missing_values_X
X.sort_values(by='GrLivArea', ascending=False)[:2]
X = X.drop(1298)
X = X.drop(523)
indexNames = X[X['GarageCars'] == 4].index
X = X.drop(indexNames)
sns.pairplot(X)
X = dataset[['FullBath', 'OverallQual', 'YearBuilt', 'TotalBsmtSF', 'GrLivArea', 'GarageCars']]
y = dataset[['SalePrice']]
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()