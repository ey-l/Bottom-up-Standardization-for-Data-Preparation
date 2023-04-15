import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
TRAIN_PATH = '_data/input/house-prices-advanced-regression-techniques/train.csv'
TEST_PATH = '_data/input/house-prices-advanced-regression-techniques/test.csv'
train_data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)
print('Shape of Train Data is ', train_data.shape)
print('Shape of Test Data is ', test_data.shape)
train_data.head()
test_data.head()
train_data.isna().sum()[train_data.isna().sum() > 0]
train_data.fillna('Unknown', inplace=True)
test_data.isna().sum()[test_data.isna().sum() > 0]
test_data.fillna('Unknown', inplace=True)
print('Numeric Type Columns - Train Data\n')
print(list(train_data._get_numeric_data().columns), '\n\n')
print('Categorical Type Columns - Train Data\n')
print(list(set(train_data.columns) - set(train_data._get_numeric_data().columns)), '\n\n')
print('Numeric Type Columns - Test Data\n')
print(list(test_data._get_numeric_data().columns), '\n\n')
print('Categorical Type Columns - Test Data\n')
print(list(set(test_data.columns) - set(test_data._get_numeric_data().columns)), '\n\n')
print('Training Data Description\n')
train_data.describe().transpose()
print('Testing Data Description\n')
test_data.describe().transpose()
train_data_numeric = list(train_data._get_numeric_data().columns)
train_data_category = list(set(train_data.columns) - set(train_data._get_numeric_data().columns))
test_data_numeric = list(test_data._get_numeric_data().columns)
test_data_category = list(set(test_data.columns) - set(test_data._get_numeric_data().columns))
oe = OrdinalEncoder()
for col in train_data_category:
    train_data[col] = oe.fit_transform(np.asarray(train_data[col].astype('str')).reshape(-1, 1))
for col in test_data_category:
    test_data[col] = oe.fit_transform(np.asarray(test_data[col].astype('str')).reshape(-1, 1))
train_data.head()
test_data.head()
l = list(set(train_data._get_numeric_data().columns))
for col in l:
    if col == 'Id':
        continue
    upper_limit = int(train_data[col].mean() + 3 * train_data[col].std())
    lower_limit = int(train_data[col].mean() - 3 * train_data[col].std())
    train_data[col] = np.where(train_data[col] > upper_limit, upper_limit, np.where(train_data[col] < lower_limit, lower_limit, train_data[col]))
for col in l:
    plt.figure(figsize=(10, 1))
    sns.boxplot(data=train_data[l], x=train_data[col], orient='h')
l = list(set(test_data._get_numeric_data().columns))
for col in l:
    if col == 'Id':
        continue
    upper_limit = int(test_data[col].mean() + 3 * test_data[col].std())
    lower_limit = int(test_data[col].mean() - 3 * test_data[col].std())
    test_data[col] = np.where(test_data[col] > upper_limit, upper_limit, np.where(test_data[col] < lower_limit, lower_limit, test_data[col]))
for col in l:
    plt.figure(figsize=(10, 1))
    sns.boxplot(data=test_data[l], x=test_data[col], orient='h')
train_data.head()
test_data.head()
train_data.corr()
test_data.corr()
from sklearn.neural_network import MLPRegressor
X = train_data.iloc[:, 0:-1]
y = train_data.loc[0:1459, 'SalePrice']
model = MLPRegressor(activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant')