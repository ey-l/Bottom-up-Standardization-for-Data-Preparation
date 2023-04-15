import numpy as np
import pandas as pd
df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
import seaborn as sns
sns.heatmap(df.isnull(), cbar=False, cmap='PuBu')
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
missing = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing.head(20)
df.drop(['Alley'], axis=1, inplace=True)
df.drop(['PoolQC'], axis=1, inplace=True)
df.drop(['MiscFeature'], axis=1, inplace=True)
df.drop(['Fence'], axis=1, inplace=True)
df.drop(['FireplaceQu'], axis=1, inplace=True)
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean())
df['GarageCond'] = df['GarageCond'].fillna(df['GarageCond'].mode()[0])
df['GarageType'] = df['GarageType'].fillna(df['GarageType'].mode()[0])
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['GarageYrBlt'].mean())
df['GarageFinish'] = df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])
df['GarageQual'] = df['GarageQual'].fillna(df['GarageQual'].mode()[0])
df['BsmtExposure'] = df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])
df['BsmtFinType2'] = df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])
df['BsmtFinType1'] = df['BsmtFinType1'].fillna(df['BsmtFinType1'].mode()[0])
df['BsmtCond'] = df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
df['BsmtQual'] = df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
df['MasVnrArea'] = df['MasVnrArea'].fillna(df['MasVnrArea'].mean())
df['MasVnrType'] = df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
from sklearn.preprocessing import LabelEncoder
lencoders = {}
for col in df.select_dtypes(include=['object']).columns:
    lencoders[col] = LabelEncoder()
    df[col] = lencoders[col].fit_transform(df[col])
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
x_t = df.drop('SalePrice', axis=1)
y_t = df['SalePrice']
clf_1 = SelectFromModel(RandomForestClassifier(n_estimators=100, max_features='log2', max_depth=4))
clf_2 = SelectFromModel(RandomForestClassifier(n_estimators=100, max_features='auto', max_depth=4))