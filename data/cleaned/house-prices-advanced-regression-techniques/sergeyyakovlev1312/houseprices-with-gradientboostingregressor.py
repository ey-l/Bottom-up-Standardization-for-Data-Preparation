import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
sample_submission = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')



y_train = train['SalePrice']
train = train.drop(['Id', 'SalePrice'], axis=1)
test = test.drop(['Id'], axis=1)
X = pd.concat([train, test], ignore_index=True)

X.info()
print(pd.unique(X['Street']))
print(pd.unique(X['Alley']))
print(pd.unique(X['FireplaceQu']))
print(pd.unique(X['Fence']))
print(pd.unique(X['MiscFeature']))

X['Alley'] = X['Alley'].fillna('No')
X['FireplaceQu'] = X['FireplaceQu'].fillna('No')
X['PoolQC'] = X['PoolQC'].fillna('No')
X['Fence'] = X['Fence'].fillna('No')
X['MiscFeature'] = X['MiscFeature'].fillna('No')

cols = []
for col in X.columns:
    if X[col].isnull().sum() > 0:
        cols.append(col)
print(cols)
cols1 = cols[:int(len(cols) / 2)]
cols2 = cols[int(len(cols) / 2):]


for col in cols:
    if col in ['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'GarageArea', 'BsmtUnfSF', 'TotalBsmtSF']:
        X[col] = X[col].fillna(X[col].mean())
    if col in ['BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt', 'GarageCars']:
        X[col] = X[col].fillna(X[col].median())
    else:
        X[col] = X[col].fillna(X[col].mode()[0])
X.info()

cols = []
for col in X.columns:
    if X[col].dtype == 'object':
        cols.append(col)
print(cols)
LE = LabelEncoder()
for col in cols:
    X[col] = LE.fit_transform(X[col])

fig = plt.figure(figsize=(20, 70))
for i in range(1, X.shape[1] + 1):
    ax = fig.add_subplot(int(X.shape[1] / 4) + 1, 4, i)
    ax.set_title(X.columns[i - 1])
    ax.set_ylabel('number of samples')
    ax.hist(X[X.columns[i - 1]], bins=100, ec='black', color='b')

X = X.drop(['Street', 'Utilities', 'Condition2', 'RoofMatl', 'BsmtFinSF2', 'Heating', 'LowQualFinSF', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'MiscFeature', 'MiscVal'], axis=1)
X_train = X.iloc[:train.shape[0], :]
X_test = X.iloc[train.shape[0]:, :]


fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Histogram')
ax.set_xlabel('price')
ax.set_ylabel('number_of_houses')
ax.hist(y_train, bins=50, ec='black', color='m')

print('Mean:', y_train.mean())
print('Variance:', y_train.var())
print('Median:', y_train.median())
print('Min:', y_train.min())
print('Max:', y_train.max())
y_train = np.log(y_train)
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Histogram')
ax.set_xlabel('log_price')
ax.set_ylabel('number_of_houses')
ax.hist(y_train, bins=50, ec='black', color='m')

print('Mean:', y_train.mean())
print('Variance:', y_train.var())
print('Median:', y_train.median())
print('Min:', y_train.min())
print('Max:', y_train.max())
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Histogram')
ax.set_xlabel('log_price')
ax.set_ylabel('number_of_houses')
ax.hist(y_train, bins=50, ec='black', color='m')
x = np.linspace(10.5, 13.5, 100)
y = 250 * (1 / (2 * np.pi * y_train.var() ** 0.5)) * np.exp(-(x - y_train.mean()) ** 2 / (2 * y_train.var()))
ax.plot(x, y, color='r')

(X_train, X_val, y_train, y_val) = train_test_split(X_train, y_train, test_size=0.05, random_state=0)

