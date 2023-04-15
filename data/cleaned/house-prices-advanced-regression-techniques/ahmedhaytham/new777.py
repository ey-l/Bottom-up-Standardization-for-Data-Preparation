import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
submission = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
print(len(train_data.columns))
print(len(test_data.columns))
train_data.head()
print(train_data.isnull().sum().sort_values(ascending=False))
train_data = train_data.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'Fence'])
train_data.describe()
categorical_data = train_data.select_dtypes(['object']).columns
train_data[categorical_data] = train_data[categorical_data].fillna(train_data[categorical_data].mode().iloc[0])
train_data[categorical_data].mode()
print(train_data.isnull().sum().sort_values(ascending=False))
trai = train_data.drop('Id', axis=1)
numerical_data = trai.select_dtypes(['float64', 'int64']).columns
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=15)
for i in numerical_data:
    train_data[i] = imputer.fit_transform(train_data[[i]])
train_data.hist(figsize=(20, 20), bins=20)

category_columns = train_data.select_dtypes(['object']).columns
print(category_columns)
train_data[category_columns] = train_data[category_columns].astype('category').apply(lambda x: x.cat.codes)
float_columns = train_data.select_dtypes(['float64']).columns
print(float_columns)
train_data['LotFrontage'] = pd.to_numeric(train_data['LotFrontage'], errors='coerce')
train_data['MasVnrArea'] = pd.to_numeric(train_data['MasVnrArea'], errors='coerce')
train_data['GarageYrBlt'] = pd.to_numeric(train_data['GarageYrBlt'], errors='coerce')
train_data['SalePrice'] = pd.to_numeric(train_data['SalePrice'], errors='coerce')
train_data = train_data.astype('float64')
sns.displot(train_data['SalePrice'])
correlation_matrix = train_data.corr()
correlation_matrix['SalePrice'].sort_values(ascending=False)
correlation_num = 30
correlation_cols = correlation_matrix.nlargest(correlation_num, 'SalePrice')['SalePrice'].index
correlation_mat_sales = np.corrcoef(train_data[correlation_cols].values.T)
sns.set(font_scale=1.25)
(f, ax) = plt.subplots(figsize=(12, 9))
hm = sns.heatmap(correlation_mat_sales, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 7}, yticklabels=correlation_cols.values, xticklabels=correlation_cols.values)

y = train_data['SalePrice']
x = train_data.drop(columns=['SalePrice', 'Id'])
print(len(x.columns))
from sklearn.model_selection import train_test_split
(X_train, X_test, Y_train, Y_test) = train_test_split(x, y, test_size=0.3, random_state=60, shuffle=True)
print(len(X_train))
print(len(X_test))
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import sklearn.metrics as sm
forest_model = RandomForestRegressor(n_estimators=150, random_state=42)