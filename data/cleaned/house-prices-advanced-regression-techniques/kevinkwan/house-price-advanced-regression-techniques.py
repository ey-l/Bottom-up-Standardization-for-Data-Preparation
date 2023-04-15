import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import itertools
data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
data.head()
data.info()
data.describe().T
plt.figure(figsize=(24, 12))
sns.heatmap(data.isnull(), cmap='mako')
columns_to_drop = []
columns_to_model = []
non_sparse_column_percentage = 90
for col in data.columns:
    if data[data[col].notnull()][col].count() / len(data) * 100 != 100:
        print(f'{col}: {data[data[col].notnull()][col].count() / len(data) * 100:.4f}%')
        if data[data[col].notnull()][col].count() / len(data) * 100 < non_sparse_column_percentage:
            columns_to_drop.append(col)
        else:
            columns_to_model.append(col)
    else:
        columns_to_model.append(col)
data = data[columns_to_model]
for col in columns_to_model:
    data = data[data[col].notnull()]
plt.figure(figsize=(24, 12))
sns.heatmap(data.isnull(), cmap='mako')
data
plt.figure(figsize=(24, 12))
sns.heatmap(test_data.isnull(), cmap='mako')
for col in test_data.columns:
    if test_data[test_data[col].notnull()][col].count() / len(test_data) * 100 != 100:
        print(f'{col}: {test_data[test_data[col].notnull()][col].count() / len(test_data) * 100:.4f}%')
plt.figure(figsize=(24, 12))
sns.heatmap(test_data[[col for col in columns_to_model if col != 'SalePrice']].isnull(), cmap='mako')
numerical_features = []
categorical_features = []
for col in data:
    if (data[col].dtype == int) | (data[col].dtype == float):
        numerical_features.append(col)
    else:
        categorical_features.append(col)
print(f'Numerical features: {numerical_features}')
print(f'Categorical features {categorical_features}')
for feature in categorical_features:
    print(f"{feature}: (Unique Count = {len(data[feature].unique())})\n\n{data[feature].unique()}\n\n{'*' * 75}")
categorical_numerical_features = []
for feature in categorical_numerical_features:
    numerical_features.remove(feature)
    catergorical_features.append(feature)
print(f'Numerical features:\n {numerical_features}\n')
print(f'Categorical features\n {categorical_features}')
plt.figure(figsize=(24, 12))
sns.heatmap(data.corr(), cmap='coolwarm', annot=True)
data.corr()[data.corr()['SalePrice'] > 0][['SalePrice']].sort_values(by='SalePrice', ascending=False)
plt.figure(figsize=(15, 5))
sns.scatterplot(data=data, x='SalePrice', y='OverallQual', hue='GrLivArea')
plt.figure(figsize=(15, 5))
sns.scatterplot(data=data, x='SalePrice', y='GrLivArea', hue='OverallQual')
plt.figure(figsize=(15, 5))
sns.scatterplot(data=data, x='SalePrice', y='GarageCars')
plt.figure(figsize=(15, 5))
sns.scatterplot(data=data, x='SalePrice', y='GarageArea', hue='GarageCars')
plt.figure(figsize=(15, 5))
sns.scatterplot(data=data, x='SalePrice', y='1stFlrSF', hue='GarageCars')
plt.figure(figsize=(15, 5))
sns.scatterplot(data=data, x='SalePrice', y='TotalBsmtSF', hue='GarageCars')
plt.figure(figsize=(15, 5))
sns.scatterplot(data=data, x='SalePrice', y='FullBath')
plt.figure(figsize=(15, 5))
sns.scatterplot(data=data, x='SalePrice', y='TotRmsAbvGrd', hue='GarageCars')
plt.figure(figsize=(15, 5))
sns.scatterplot(data=data, x='SalePrice', y='YearBuilt', hue='GarageCars')
plt.figure(figsize=(15, 5))
sns.scatterplot(data=data, x='SalePrice', y='YearRemodAdd', hue='GarageCars')
plt.figure(figsize=(15, 5))
sns.scatterplot(data=data, x='SalePrice', y='GarageYrBlt', hue='GarageCars')
plt.figure(figsize=(15, 5))
sns.scatterplot(data=data, x='SalePrice', y='MasVnrArea', hue='GarageCars')
plt.figure(figsize=(15, 5))
sns.scatterplot(data=data, x='SalePrice', y='Fireplaces', hue='GarageCars')
plt.figure(figsize=(15, 5))
sns.scatterplot(data=data, x='SalePrice', y='BsmtFinSF1', hue='GarageCars')
plt.figure(figsize=(15, 5))
sns.scatterplot(data=data, x='SalePrice', y='OpenPorchSF', hue='GarageCars')
plt.figure(figsize=(15, 5))
sns.scatterplot(data=data, x='SalePrice', y='2ndFlrSF', hue='GarageCars')
plt.figure(figsize=(15, 5))
sns.scatterplot(data=data, x='SalePrice', y='WoodDeckSF', hue='GarageCars')
plt.figure(figsize=(15, 5))
sns.scatterplot(data=data, x='SalePrice', y='HalfBath', hue='GarageCars')
plt.figure(figsize=(15, 5))
sns.scatterplot(data=data, x='SalePrice', y='LotArea', hue='GarageCars')
data = data[numerical_features]
X = data.drop(['Id', 'SalePrice'], axis=1)
y = data['SalePrice']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=101)
lm = LinearRegression()