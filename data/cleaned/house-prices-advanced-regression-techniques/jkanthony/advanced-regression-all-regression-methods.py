import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
(train.shape, test.shape)
train.head()
df = pd.concat((train, test))
df.shape
df.head()
df.info()
df.describe()
df.select_dtypes(include=['int64', 'float64']).columns
df.select_dtypes(include=['object']).columns
plt.figure(figsize=(16, 9))
sns.heatmap(df.isnull())
null_percent = df.isnull().sum() / df.shape[0] * 100
null_percent
col_for_drop = null_percent[null_percent > 20].keys()
df = df.drop(col_for_drop, 'columns')
df.shape
for i in df.columns:
    print(i + '\t' + str(len(df[i].unique())))
plt.figure(figsize=(10, 8))
bar = sns.distplot(train['SalePrice'])
bar.legend(['Skewness: {:.2f}'.format(train['SalePrice'].skew())])
plt.figure(figsize=(25, 25))
ax = sns.heatmap(train.corr(), cmap='coolwarm', annot=True, linewidth=2)
(bottom, top) = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
hig_corr = train.corr()
hig_corr_features = hig_corr.index[abs(hig_corr['SalePrice']) >= 0.5]
print(hig_corr_features)
plt.figure(figsize=(10, 8))
ax = sns.heatmap(train[hig_corr_features].corr(), cmap='coolwarm', annot=True, linewidth=3)
(bottom, top) = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.figure(figsize=(16, 9))
for i in range(len(hig_corr_features)):
    if i <= 9:
        plt.subplot(3, 4, i + 1)
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        sns.regplot(data=train, x=hig_corr_features[i], y='SalePrice')
df.isnull().sum()
df['LotFrontage'].fillna(np.mean(df['LotFrontage']), inplace=True)

def fill_null(values):
    type = values[0]
    area = values[1]
    if pd.isnull(type):
        return ('None', 0)
    else:
        return values
df[['MasVnrType', 'MasVnrArea']] = df[['MasVnrType', 'MasVnrArea']].apply(fill_null, axis=1)
df['BsmtQual'].fillna(df['BsmtQual'].mode()[0], inplace=True)
df['BsmtCond'].fillna(df['BsmtCond'].mode()[0], inplace=True)
df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0], inplace=True)
df['BsmtFinType1'].fillna(df['BsmtFinType1'].mode()[0], inplace=True)
df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0], inplace=True)
df['Electrical'].fillna(df['Electrical'].mode()[0], inplace=True)
df['GarageType'].fillna('No', inplace=True)
df['GarageYrBlt'].fillna(0, inplace=True)
df['GarageFinish'].fillna('No', inplace=True)
df['GarageQual'].fillna('No', inplace=True)
df.drop('GarageCond', axis=1, inplace=True)
df.drop('Exterior2nd', axis=1, inplace=True)
df['MSZoning'].fillna(df['MSZoning'].mode()[0], inplace=True)
df['Utilities'].fillna(df['Utilities'].mode()[0], inplace=True)
df['Exterior1st'].fillna(df['Exterior1st'].mode()[0], inplace=True)
df['BsmtFinSF1'].fillna(df['BsmtFinSF1'].mean(), inplace=True)
df['BsmtFinSF2'].fillna(df['BsmtFinSF2'].mean(), inplace=True)
df['BsmtUnfSF'].fillna(df['BsmtUnfSF'].mean(), inplace=True)
df['TotalBsmtSF'].fillna(df['TotalBsmtSF'].mean(), inplace=True)
df['BsmtFullBath'].fillna(df['BsmtFullBath'].mode()[0], inplace=True)
df['BsmtHalfBath'].fillna(df['BsmtHalfBath'].mode()[0], inplace=True)
df['KitchenQual'].fillna(df['KitchenQual'].mode()[0], inplace=True)
df['Functional'].fillna(df['Functional'].mode()[0], inplace=True)
df['GarageCars'].fillna(df['GarageCars'].mode()[0], inplace=True)
df['GarageArea'].fillna(df['GarageArea'].mean(), inplace=True)
df['SaleType'].fillna(df['SaleType'].mode()[0], inplace=True)
df.isnull().sum()
object_columns = df.select_dtypes(include=['object']).columns
object_columns
for i in object_columns:
    df = pd.get_dummies(df, columns=[i])
df.info()
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()