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
train_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
train_df.head()
test_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
test_df.head()
plt.figure(figsize=(20, 5))
sns.distplot(train_df.SalePrice)
plt.title('Sales Price distribution in Train dataset')
plt.ylabel('Density')
isna_train = train_df.isnull().sum().sort_values(ascending=False)
isna_test = test_df.isnull().sum().sort_values(ascending=False)
plt.subplot(2, 1, 1)
plt_1 = isna_train[:30].plot(kind='bar')
plt.ylabel('Train Data')
plt.subplot(2, 1, 2)
isna_test[:30].plot(kind='bar')
plt.ylabel('Test Data')
plt.xlabel('Number of features which are NaNs')
(train_df.isnull().sum() / len(train_df)).sort_values(ascending=False)[:25]
missing_percentage = (train_df.isnull().sum() / len(train_df)).sort_values(ascending=False)[:20]
missing_percentage
train_df = train_df.drop(missing_percentage.index[:5], 1)
test_df = test_df.drop(missing_percentage.index[:5], 1)
cols = train_df[missing_percentage.index[5:]].columns
num_cols = train_df[missing_percentage.index[5:]]._get_numeric_data().columns
print('Numerical Columns', num_cols)
cat_cols = list(set(cols) - set(num_cols))
print('Categorical Columns:', cat_cols)
plt.figure(figsize=[12, 10])
plt.subplot(331)
sns.distplot(train_df['LotFrontage'].dropna().values)
plt.xlabel('LotFrontage')
plt.subplot(332)
sns.distplot(train_df['GarageYrBlt'].dropna().values)
plt.xlabel('GarageYrBlt')
plt.subplot(333)
sns.distplot(train_df['MasVnrArea'].dropna().values)
plt.xlabel('MasVnrArea')
plt.suptitle('Distribution of numerical data before Data imputaion')
train_df['LotFrontage'] = train_df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
train_df['GarageYrBlt'] = train_df.groupby('Neighborhood')['GarageYrBlt'].transform(lambda x: x.fillna(x.median()))
train_df['MasVnrArea'] = train_df.groupby('Neighborhood')['MasVnrArea'].transform(lambda x: x.fillna(x.median()))
plt.figure(figsize=[12, 10])
plt.subplot(331)
sns.distplot(train_df['LotFrontage'].values)
plt.xlabel('LotFrontage')
plt.subplot(332)
sns.distplot(train_df['GarageYrBlt'].values)
plt.xlabel('GarageBlt')
plt.subplot(333)
sns.distplot(train_df['MasVnrArea'].values)
plt.xlabel('MasVnrArea')
plt.suptitle('Distribution of data after data imputaion')
test_df['LotFrontage'] = train_df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
test_df['GarageYrBlt'] = train_df.groupby('Neighborhood')['GarageYrBlt'].transform(lambda x: x.fillna(x.median()))
test_df['MasVnrArea'] = train_df.groupby('Neighborhood')['MasVnrArea'].transform(lambda x: x.fillna(x.median()))
for column in cat_cols:
    train_df[column] = train_df.groupby('Neighborhood')[column].transform(lambda x: x.fillna(x.mode()))
    test_df[column] = test_df.groupby('Neighborhood')[column].transform(lambda x: x.fillna(x.mode()))
num_cols = train_df._get_numeric_data().columns
print('Numerical Columns', num_cols)
cat_cols = list(set(cols) - set(num_cols))
print('Categorical Columns:', cat_cols)
Neighbor = train_df.groupby(['Neighborhood', 'YearBuilt'])['SalePrice']
Neighbor = Neighbor.describe()['mean'].to_frame()
Neighbor = Neighbor.reset_index(level=[0, 1])
Neighbor = Neighbor.groupby('Neighborhood')
Neighbor_index = train_df['Neighborhood'].unique()
fig = plt.figure(figsize=(50, 12))
fig.suptitle('Yearwise Trend of each Neighborhood')
for num in range(1, 25):
    temp = Neighbor.get_group(Neighbor_index[num])
    ax = fig.add_subplot(6, 4, num)
    ax.plot(temp['YearBuilt'], temp['mean'])
    ax.set_title(temp['Neighborhood'].unique())
cols = train_df.columns
num_cols = train_df._get_numeric_data().columns
print('Numerical Columns', num_cols)
cat_cols = list(set(cols) - set(num_cols))
print('Categorical Columns:', cat_cols)
from sklearn.preprocessing import LabelEncoder
for i in cat_cols:
    train_df[i] = LabelEncoder().fit_transform(train_df[i].astype(str))
    test_df[i] = LabelEncoder().fit_transform(test_df[i].astype(str))
(fig, ax) = plt.subplots(figsize=(10, 10))
sns.heatmap(train_df.corr(), ax=ax, annot=False, linewidth=0.02, linecolor='black', fmt='.2f', cmap='Blues')

corr = train_df.corr()
corr = corr.sort_values(by=['SalePrice'], ascending=False).iloc[0].sort_values(ascending=False)
plt.figure(figsize=(15, 20))
sns.barplot(x=corr.values, y=corr.index.values)
plt.title('Correlation Plot')
index = []
Train = pd.DataFrame()
Y = train_df['SalePrice']
for i in range(0, len(corr)):
    if corr[i] > 0.15 and corr.index[i] != 'SalePrice':
        index.append(corr.index[i])
X = train_df[index]
X['cond*qual'] = train_df['OverallCond'] * train_df['OverallQual'] / 100.0
X['home_age_when_sold'] = train_df['YrSold'] - train_df['YearBuilt']
X['garage_age_when_sold'] = train_df['YrSold'] - train_df['GarageYrBlt']
X['TotalSF'] = train_df['TotalBsmtSF'] + train_df['1stFlrSF'] + train_df['2ndFlrSF']
X['total_porch_area'] = train_df['WoodDeckSF'] + train_df['OpenPorchSF'] + train_df['EnclosedPorch'] + train_df['3SsnPorch'] + train_df['ScreenPorch']
X['Totalsqrfootage'] = train_df['BsmtFinSF1'] + train_df['BsmtFinSF2'] + train_df['1stFlrSF'] + train_df['2ndFlrSF']
X['Total_Bathrooms'] = train_df['FullBath'] + 0.5 * train_df['HalfBath'] + train_df['BsmtFullBath'] + 0.5 * train_df['BsmtHalfBath']
test_df['cond*qual'] = test_df['OverallCond'] * test_df['OverallQual'] / 100.0
test_df['home_age_when_sold'] = test_df['YrSold'] - test_df['YearBuilt']
test_df['garage_age_when_sold'] = test_df['YrSold'] - test_df['GarageYrBlt']
test_df['TotalSF'] = test_df['TotalBsmtSF'] + test_df['1stFlrSF'] + test_df['2ndFlrSF']
test_df['total_porch_area'] = test_df['WoodDeckSF'] + test_df['OpenPorchSF'] + test_df['EnclosedPorch'] + test_df['3SsnPorch'] + test_df['ScreenPorch']
test_df['Totalsqrfootage'] = test_df['BsmtFinSF1'] + test_df['BsmtFinSF2'] + test_df['1stFlrSF'] + test_df['2ndFlrSF']
test_df['Total_Bathrooms'] = test_df['FullBath'] + 0.5 * test_df['HalfBath'] + test_df['BsmtFullBath'] + 0.5 * test_df['BsmtHalfBath']
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