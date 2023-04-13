import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1.head()
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input0.head()
plt.figure(figsize=(20, 5))
sns.distplot(_input1.SalePrice)
plt.title('Sales Price distribution in Train dataset')
plt.ylabel('Density')
isna_train = _input1.isnull().sum().sort_values(ascending=False)
isna_test = _input0.isnull().sum().sort_values(ascending=False)
plt.subplot(2, 1, 1)
plt_1 = isna_train[:30].plot(kind='bar')
plt.ylabel('Train Data')
plt.subplot(2, 1, 2)
isna_test[:30].plot(kind='bar')
plt.ylabel('Test Data')
plt.xlabel('Number of features which are NaNs')
(_input1.isnull().sum() / len(_input1)).sort_values(ascending=False)[:25]
missing_percentage = (_input1.isnull().sum() / len(_input1)).sort_values(ascending=False)[:20]
missing_percentage
_input1 = _input1.drop(missing_percentage.index[:5], 1)
_input0 = _input0.drop(missing_percentage.index[:5], 1)
cols = _input1[missing_percentage.index[5:]].columns
num_cols = _input1[missing_percentage.index[5:]]._get_numeric_data().columns
print('Numerical Columns', num_cols)
cat_cols = list(set(cols) - set(num_cols))
print('Categorical Columns:', cat_cols)
plt.figure(figsize=[12, 10])
plt.subplot(331)
sns.distplot(_input1['LotFrontage'].dropna().values)
plt.xlabel('LotFrontage')
plt.subplot(332)
sns.distplot(_input1['GarageYrBlt'].dropna().values)
plt.xlabel('GarageYrBlt')
plt.subplot(333)
sns.distplot(_input1['MasVnrArea'].dropna().values)
plt.xlabel('MasVnrArea')
plt.suptitle('Distribution of numerical data before Data imputaion')
_input1['LotFrontage'] = _input1.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
_input1['GarageYrBlt'] = _input1.groupby('Neighborhood')['GarageYrBlt'].transform(lambda x: x.fillna(x.median()))
_input1['MasVnrArea'] = _input1.groupby('Neighborhood')['MasVnrArea'].transform(lambda x: x.fillna(x.median()))
plt.figure(figsize=[12, 10])
plt.subplot(331)
sns.distplot(_input1['LotFrontage'].values)
plt.xlabel('LotFrontage')
plt.subplot(332)
sns.distplot(_input1['GarageYrBlt'].values)
plt.xlabel('GarageBlt')
plt.subplot(333)
sns.distplot(_input1['MasVnrArea'].values)
plt.xlabel('MasVnrArea')
plt.suptitle('Distribution of data after data imputaion')
_input0['LotFrontage'] = _input1.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
_input0['GarageYrBlt'] = _input1.groupby('Neighborhood')['GarageYrBlt'].transform(lambda x: x.fillna(x.median()))
_input0['MasVnrArea'] = _input1.groupby('Neighborhood')['MasVnrArea'].transform(lambda x: x.fillna(x.median()))
for column in cat_cols:
    _input1[column] = _input1.groupby('Neighborhood')[column].transform(lambda x: x.fillna(x.mode()))
    _input0[column] = _input0.groupby('Neighborhood')[column].transform(lambda x: x.fillna(x.mode()))
num_cols = _input1._get_numeric_data().columns
print('Numerical Columns', num_cols)
cat_cols = list(set(cols) - set(num_cols))
print('Categorical Columns:', cat_cols)
Neighbor = _input1.groupby(['Neighborhood', 'YearBuilt'])['SalePrice']
Neighbor = Neighbor.describe()['mean'].to_frame()
Neighbor = Neighbor.reset_index(level=[0, 1])
Neighbor = Neighbor.groupby('Neighborhood')
Neighbor_index = _input1['Neighborhood'].unique()
fig = plt.figure(figsize=(50, 12))
fig.suptitle('Yearwise Trend of each Neighborhood')
for num in range(1, 25):
    temp = Neighbor.get_group(Neighbor_index[num])
    ax = fig.add_subplot(6, 4, num)
    ax.plot(temp['YearBuilt'], temp['mean'])
    ax.set_title(temp['Neighborhood'].unique())
cols = _input1.columns
num_cols = _input1._get_numeric_data().columns
print('Numerical Columns', num_cols)
cat_cols = list(set(cols) - set(num_cols))
print('Categorical Columns:', cat_cols)
from sklearn.preprocessing import LabelEncoder
for i in cat_cols:
    _input1[i] = LabelEncoder().fit_transform(_input1[i].astype(str))
    _input0[i] = LabelEncoder().fit_transform(_input0[i].astype(str))
(fig, ax) = plt.subplots(figsize=(10, 10))
sns.heatmap(_input1.corr(), ax=ax, annot=False, linewidth=0.02, linecolor='black', fmt='.2f', cmap='Blues')
corr = _input1.corr()
corr = corr.sort_values(by=['SalePrice'], ascending=False).iloc[0].sort_values(ascending=False)
plt.figure(figsize=(15, 20))
sns.barplot(x=corr.values, y=corr.index.values)
plt.title('Correlation Plot')
index = []
Train = pd.DataFrame()
Y = _input1['SalePrice']
for i in range(0, len(corr)):
    if corr[i] > 0.15 and corr.index[i] != 'SalePrice':
        index.append(corr.index[i])
X = _input1[index]
X['cond*qual'] = _input1['OverallCond'] * _input1['OverallQual'] / 100.0
X['home_age_when_sold'] = _input1['YrSold'] - _input1['YearBuilt']
X['garage_age_when_sold'] = _input1['YrSold'] - _input1['GarageYrBlt']
X['TotalSF'] = _input1['TotalBsmtSF'] + _input1['1stFlrSF'] + _input1['2ndFlrSF']
X['total_porch_area'] = _input1['WoodDeckSF'] + _input1['OpenPorchSF'] + _input1['EnclosedPorch'] + _input1['3SsnPorch'] + _input1['ScreenPorch']
X['Totalsqrfootage'] = _input1['BsmtFinSF1'] + _input1['BsmtFinSF2'] + _input1['1stFlrSF'] + _input1['2ndFlrSF']
X['Total_Bathrooms'] = _input1['FullBath'] + 0.5 * _input1['HalfBath'] + _input1['BsmtFullBath'] + 0.5 * _input1['BsmtHalfBath']
_input0['cond*qual'] = _input0['OverallCond'] * _input0['OverallQual'] / 100.0
_input0['home_age_when_sold'] = _input0['YrSold'] - _input0['YearBuilt']
_input0['garage_age_when_sold'] = _input0['YrSold'] - _input0['GarageYrBlt']
_input0['TotalSF'] = _input0['TotalBsmtSF'] + _input0['1stFlrSF'] + _input0['2ndFlrSF']
_input0['total_porch_area'] = _input0['WoodDeckSF'] + _input0['OpenPorchSF'] + _input0['EnclosedPorch'] + _input0['3SsnPorch'] + _input0['ScreenPorch']
_input0['Totalsqrfootage'] = _input0['BsmtFinSF1'] + _input0['BsmtFinSF2'] + _input0['1stFlrSF'] + _input0['2ndFlrSF']
_input0['Total_Bathrooms'] = _input0['FullBath'] + 0.5 * _input0['HalfBath'] + _input0['BsmtFullBath'] + 0.5 * _input0['BsmtHalfBath']
Old_Cols = ['OverallCond', 'OverallQual', 'YrSold', 'YearBuilt', 'YrSold', 'GarageYrBlt', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'BsmtFinSF1', 'BsmtFinSF2', '1stFlrSF', '2ndFlrSF', 'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
Final_cols = []
for i in X.columns:
    if i not in Old_Cols and i != 'SalePrice':
        Final_cols.append(i)
X = X[Final_cols]
fig = plt.figure(figsize=(20, 16))
plt.subplot(2, 2, 1)
plt.scatter(X['home_age_when_sold'], Y)
plt.title('Home Age Vs SalePrice ')
plt.ylabel('SalePrice')
plt.xlabel('Home Age')
plt.subplot(2, 2, 2)
plt.scatter(X['Total_Bathrooms'], Y)
plt.title('Total_Bathrooms Vs SalePrice ')
plt.ylabel('SalePrice')
plt.xlabel('Total_Bathrooms')
plt.subplot(2, 2, 3)
plt.scatter(X['TotalSF'], Y)
plt.title('TotalSF Vs SalePrice ')
plt.ylabel('SalePrice')
plt.xlabel('TotalSF')
plt.subplot(2, 2, 4)
plt.scatter(X['cond*qual'], Y)
plt.title('House Condition Vs SalePrice ')
plt.ylabel('SalePrice')
plt.xlabel('cond*qual')
X.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
temp = pd.DataFrame()
temp = X
temp['SalePrice'] = Y
for i in range(0, len(temp.columns), 5):
    sns.pairplot(data=temp, x_vars=temp.columns[i:i + 5], y_vars=['SalePrice'])
from sklearn.preprocessing import LabelEncoder
for c in Old_Cols:
    lbl = LabelEncoder()