import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train_df.info()
plt.figure(figsize=(10, 5), dpi=200)
sns.histplot(x=train_df['SalePrice'])
sns.boxplot(data=train_df, x='SalePrice')
train_df['SalePrice'].describe()
train_df.corr()['SalePrice'].sort_values(ascending=False)
plt.figure(figsize=(10, 5), dpi=200)
sns.scatterplot(x=train_df['GrLivArea'], y=train_df['SalePrice'])
train_df[train_df['GrLivArea'] > 4000]
plt.figure(figsize=(10, 5), dpi=200)
sns.scatterplot(x=train_df['OverallQual'], y=train_df['SalePrice'])
train_df[(train_df['OverallQual'] == 10) & (train_df['SalePrice'] < 200000)]
drop_ind = train_df[(train_df['OverallQual'] == 10) & (train_df['SalePrice'] < 200000)].index
train_df = train_df.drop(drop_ind, axis=0)
plt.figure(figsize=(10, 5), dpi=200)
sns.scatterplot(x=train_df['GrLivArea'], y=train_df['SalePrice'])
plt.figure(figsize=(15, 10), dpi=200)
sns.heatmap(train_df.corr(), annot=False)
test_df.info()
main_df = pd.concat([train_df, test_df], ignore_index=True)
main_df
main_df.info()
100 * main_df.isnull().sum() / len(main_df)

def percent_missing(df):
    percentage = 100 * main_df.isnull().sum() / len(main_df)
    percentage = percentage[percentage > 0]
    percentage = percentage.sort_values(ascending=False)
    return percentage
miss_data_rel_freq = percent_missing(main_df)
miss_data_rel_freq
plt.figure(figsize=(10, 5), dpi=200)
sns.barplot(x=miss_data_rel_freq.index, y=miss_data_rel_freq.values)
plt.xticks(rotation=90)

plt.figure(figsize=(10, 5), dpi=200)
sns.barplot(x=miss_data_rel_freq.index, y=miss_data_rel_freq.values)
plt.xticks(rotation=90)
plt.ylim(0, 1)

miss_data_rel_freq[miss_data_rel_freq < 1]
main_df[main_df['Electrical'].isnull()]
main_df = main_df.dropna(axis=0, subset=['Electrical'])
main_df[main_df['GarageArea'].isnull()]
main_df['GarageArea'].describe()
main_df['GarageArea'] = main_df['GarageArea'].fillna(472)
main_df[main_df['GarageCars'].isnull()]
main_df['GarageCars'].describe()
main_df['GarageCars'] = main_df['GarageCars'].fillna(2)
miss_data_rel_freq = percent_missing(main_df)
miss_data_rel_freq[miss_data_rel_freq < 1]
main_df[main_df['BsmtFullBath'].isnull()]
main_df[main_df['BsmtHalfBath'].isnull()]
main_df[main_df['BsmtUnfSF'].isnull()]
main_df[main_df['TotalBsmtSF'].isnull()]['BsmtExposure']
base_num_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']
main_df[base_num_cols] = main_df[base_num_cols].fillna(0)
base_str_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
main_df[base_str_cols] = main_df[base_str_cols].fillna('None')
miss_data_rel_freq = percent_missing(main_df)
miss_data_rel_freq[miss_data_rel_freq < 1]
plt.figure(figsize=(10, 5), dpi=200)
sns.barplot(x=miss_data_rel_freq.index, y=miss_data_rel_freq.values)
plt.xticks(rotation=90)
plt.ylim(0, 1)

main_df[main_df['SaleType'].isnull()]
main_df['SaleType'] = main_df['SaleType'].fillna('Oth')
main_df['Exterior1st'] = main_df['Exterior1st'].fillna('Other')
main_df['Exterior2nd'] = main_df['Exterior2nd'].fillna('Other')
main_df[main_df['MSZoning'].isnull()]
sns.countplot(x=main_df['MSZoning'])
main_df['MSZoning'] = main_df['MSZoning'].fillna('RL')
main_df[main_df['Functional'].isnull()]
sns.countplot(x=main_df['Functional'])
main_df['Functional'] = main_df['Functional'].fillna('Typ')
main_df[main_df['Utilities'].isnull()]
sns.countplot(x=main_df['Utilities'])
main_df['Utilities'] = main_df['Utilities'].fillna('AllPub')
main_df[main_df['KitchenQual'].isnull()]
sns.countplot(x=main_df['KitchenQual'])
main_df['KitchenQual'] = main_df['KitchenQual'].fillna('TA')
main_df[main_df['MasVnrType'].isnull()]
main_df[main_df['MasVnrArea'].isnull()]
main_df['MasVnrType'] = main_df['MasVnrType'].fillna('None')
main_df['MasVnrArea'] = main_df['MasVnrArea'].fillna(0)
miss_data_rel_freq = percent_missing(main_df)
miss_data_rel_freq
plt.figure(figsize=(10, 5), dpi=200)
sns.barplot(x=miss_data_rel_freq.index, y=miss_data_rel_freq.values)
plt.xticks(rotation=90)

gar_str_cols = ['GarageFinish', 'GarageQual', 'GarageCond', 'GarageType']
main_df[gar_str_cols] = main_df[gar_str_cols].fillna('NA')
main_df['GarageYrBlt'] = main_df['GarageYrBlt'].fillna(0)
miss_data_rel_freq = percent_missing(main_df)
miss_data_rel_freq
plt.figure(figsize=(10, 5), dpi=200)
sns.barplot(x=miss_data_rel_freq.index, y=miss_data_rel_freq.values)
plt.xticks(rotation=90)

not_available = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
main_df[not_available] = main_df[not_available].fillna('NA')
plt.figure(figsize=(10, 10), dpi=200)
sns.boxplot(x=main_df['LotFrontage'], y=main_df['Neighborhood'], orient='h')
main_df['LotFrontage'] = main_df.groupby('Neighborhood')['LotFrontage'].transform(lambda value: value.fillna(value.mean()))
miss_data_rel_freq = percent_missing(main_df)
miss_data_rel_freq
main_df['MSSubClass'] = main_df['MSSubClass'].apply(str)
categorical_df = main_df.select_dtypes(include='object')
categorical_df
numeric_df = main_df.select_dtypes(exclude='object')
numeric_df
categorical_dummies = pd.get_dummies(categorical_df, drop_first=True)
categorical_dummies
final_df = pd.concat([numeric_df, categorical_dummies], axis=1)
final_df
final_train_df = final_df.dropna()
final_train_df
final_test_df = final_df[final_df['SalePrice'].isnull()]
final_test_df
final_train_df = final_train_df.drop('Id', axis=1)
final_train_df
final_train_df.corr()['SalePrice'].sort_values(ascending=False)
x = final_train_df.drop('SalePrice', axis=1)
y = final_train_df['SalePrice']
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2, random_state=45)
(x_test, x_val, y_test, y_val) = train_test_split(x_test, y_test, test_size=0.5, random_state=45)
scaler = StandardScaler()
scaled_x_train = scaler.fit_transform(x_train)
scaled_x_test = scaler.transform(x_test)
scaled_x_val = scaler.transform(x_val)
base_elastic_model = ElasticNet()
param_grid = {'alpha': [250, 300, 400, 450, 475, 500, 525, 550], 'l1_ratio': [0.95, 0.96, 0.97, 0.98, 0.99, 1, 1.1, 1.15, 1.25, 1.5]}
grid_model = GridSearchCV(estimator=base_elastic_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=2)