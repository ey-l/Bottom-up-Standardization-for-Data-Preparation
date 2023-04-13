import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_rows', None)
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
_input1['train_flag'] = 1
_input0['train_flag'] = 0
df = pd.concat([_input1, _input0], axis=0)
print(df.head())
print('--------------------------------------')
print(df.tail())
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum() / len(df)).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
sales = df['SalePrice']
df = df.drop(missing_data[missing_data['Percent'] > 0.3].index, 1, inplace=False)
df.shape
width = 20
height = 5
plt.figure(figsize=(width, height))
sns.heatmap(df.isnull(), yticklabels=False, cbar=False)
df.shape
df = df.fillna(df.agg(['median', lambda x: x.value_counts().index[0]]).ffill().iloc[-1, :], inplace=False)
df = pd.concat([df, sales], axis=1)
plt.figure(figsize=(width, height))
sns.heatmap(df.isnull(), yticklabels=False, cbar=False)
num_cols = list(df.select_dtypes(exclude=['object']).columns)
for val in ['OverallQual', 'OverallCond', 'MSSubClass', 'YearBuilt', 'YearRemodAdd', 'YrSold']:
    num_cols.remove(val)
cat_cols = list(df.select_dtypes(include=['object']).columns)
cat_cols.extend(['OverallQual', 'OverallCond', 'MSSubClass', 'YearBuilt', 'YearRemodAdd', 'YrSold'])
df.columns
list(num_cols)
for col in num_cols:
    (fig, ax) = plt.subplots(1, 3, figsize=(10, 4), tight_layout=True)
    sns.histplot(data=df.iloc[:1459, :], x=col, ax=ax[0])
    sns.boxplot(data=df.iloc[:1459, :], x=col, ax=ax[1])
    sns.scatterplot(data=df.iloc[:1459, :], x=col, y='SalePrice', ax=ax[2])
for col in cat_cols:
    (fig, ax) = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)
    sns.countplot(data=df, x=col, ax=ax[0])
    sns.boxplot(data=df, x=col, y='SalePrice', ax=ax[1])
corrmat = df.corr()
(f, ax) = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)
columns = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition2', 'BldgType', 'Condition1', 'HouseStyle', 'SaleType', 'SaleCondition', 'ExterCond', 'ExterQual', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive']

def category_onehot_multcols(multcolumns):
    final_df = df.copy()
    df_final = final_df
    i = 0
    for fields in multcolumns:
        print(fields)
        df1 = pd.get_dummies(final_df[fields], drop_first=True)
        final_df = final_df.drop([fields], axis=1, inplace=False)
        if i == 0:
            df_final = df1.copy()
        else:
            df_final = pd.concat([df_final, df1], axis=1)
        i = i + 1
    df_final = pd.concat([final_df, df_final], axis=1)
    return df_final
main_df = df.copy()
df.head()
df = category_onehot_multcols(columns)
df = df.loc[:, ~df.columns.duplicated()]
df_Train = df.iloc[:1459, :]
df_Test = df.iloc[1460:, :]
X_train = df_Train.drop(['SalePrice'], axis=1)
y_train = df_Train['SalePrice']
import xgboost
regressor = xgboost.XGBRegressor()