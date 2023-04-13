import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1.head()
_input1.columns
_input1.describe()
_input1 = _input1.drop('Id', axis=1, inplace=False)
_input1.head()
_input1 = _input1.rename(columns={'BedroomAbvGr': 'Bedroom', 'KitchenAbvGr': 'Kitchen'}, inplace=False)
_input1[['Bedroom', 'Kitchen']].head()
_input1['Bedroom'].value_counts()
_input1.isnull().sum()[_input1.isnull().sum() > 0]
_input1 = _input1.drop(['MiscFeature', 'Alley', 'PoolQC', 'Fence'], axis=1, inplace=False)
_input1['Fireplaces'].value_counts()
_input1 = _input1.drop(['FireplaceQu'], axis=1, inplace=False)
_input1['LotFrontage'].value_counts()
_input1 = _input1.drop(['LotFrontage'], axis=1, inplace=False)
_input1.isnull().sum()[_input1.isnull().sum() > 0]
_input1['Electrical'] = _input1['Electrical'].fillna(_input1['Electrical'].mode()[0], inplace=False)
_input1['Electrical'].isnull().sum()
_input1['MasVnrType'].value_counts()
_input1['MasVnrType'] = _input1['MasVnrType'].fillna(_input1['MasVnrType'].mode()[0], inplace=False)
_input1['MasVnrArea'] = _input1['MasVnrArea'].fillna(_input1['MasVnrArea'].mode()[0], inplace=False)
_input1.isnull().sum()[_input1.isnull().sum() > 0]
_input1.corr()['SalePrice'].sort_values(ascending=False)
_input1[['Bedroom', 'SalePrice']].corr()
sns.regplot(x='Bedroom', y='SalePrice', data=_input1)
sns.regplot(x='Kitchen', y='SalePrice', data=_input1)
_input1[_input1['Kitchen'] == 0]
sns.regplot(x='OverallQual', y='SalePrice', data=_input1)
sns.regplot(x='GarageCars', y='SalePrice', data=_input1)
lm = LinearRegression()
features = ['OverallQual', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd']