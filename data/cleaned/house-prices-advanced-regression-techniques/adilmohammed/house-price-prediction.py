import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df.head()
df.columns
df.describe()
df.drop('Id', axis=1, inplace=True)
df.head()
df.rename(columns={'BedroomAbvGr': 'Bedroom', 'KitchenAbvGr': 'Kitchen'}, inplace=True)
df[['Bedroom', 'Kitchen']].head()
df['Bedroom'].value_counts()
df.isnull().sum()[df.isnull().sum() > 0]
df.drop(['MiscFeature', 'Alley', 'PoolQC', 'Fence'], axis=1, inplace=True)
df['Fireplaces'].value_counts()
df.drop(['FireplaceQu'], axis=1, inplace=True)
df['LotFrontage'].value_counts()
df.drop(['LotFrontage'], axis=1, inplace=True)
df.isnull().sum()[df.isnull().sum() > 0]
df['Electrical'].fillna(df['Electrical'].mode()[0], inplace=True)
df['Electrical'].isnull().sum()
df['MasVnrType'].value_counts()
df['MasVnrType'].fillna(df['MasVnrType'].mode()[0], inplace=True)
df['MasVnrArea'].fillna(df['MasVnrArea'].mode()[0], inplace=True)
df.isnull().sum()[df.isnull().sum() > 0]
df.corr()['SalePrice'].sort_values(ascending=False)
df[['Bedroom', 'SalePrice']].corr()
sns.regplot(x='Bedroom', y='SalePrice', data=df)
sns.regplot(x='Kitchen', y='SalePrice', data=df)
df[df['Kitchen'] == 0]
sns.regplot(x='OverallQual', y='SalePrice', data=df)
sns.regplot(x='GarageCars', y='SalePrice', data=df)
lm = LinearRegression()
features = ['OverallQual', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd']