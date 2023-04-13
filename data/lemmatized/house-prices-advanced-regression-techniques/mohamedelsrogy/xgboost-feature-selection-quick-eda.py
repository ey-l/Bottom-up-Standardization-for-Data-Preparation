import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
_input1 = _input1.drop('Id', axis=1, inplace=False)
percent_missing = _input1.isnull().sum() * 100 / len(_input1)
missing_value_df = pd.DataFrame({'column_name': _input1.columns, 'percent_missing': percent_missing})
missing_value_df = missing_value_df[missing_value_df.percent_missing > 0]
plt.figure(figsize=(15, 10))
plt.barh(missing_value_df['column_name'], missing_value_df['percent_missing'], color='darkblue')
plt.title('The Percentages Of The Columns Null Values', fontsize=15)
_input1.duplicated().sum()
_input1 = _input1.drop(['Fence', 'MiscFeature', 'PoolQC', 'FireplaceQu', 'Alley'], axis=1, inplace=False)
_input1.shape
percent_missing = _input1.isnull().sum() * 100 / len(_input1)
missing_value_df = pd.DataFrame({'column_name': _input1.columns, 'percent_missing': percent_missing})
missing_value_df = missing_value_df[missing_value_df.percent_missing > 0]
plt.figure(figsize=(10, 8))
_input1.boxplot(['LotFrontage', 'MasVnrArea', 'GarageYrBlt'])
plt.title('The Box Plot For the Floating Null Columns', fontsize=15)
_input1[missing_value_df.column_name.values].describe()
_input1.MasVnrArea = _input1.MasVnrArea.fillna(method='ffill', inplace=False)
_input1.MasVnrArea = _input1.MasVnrArea.fillna(method='bfill', inplace=False)
_input1.LotFrontage = _input1.LotFrontage.fillna(_input1.LotFrontage.median(), inplace=False)
_input1.GarageYrBlt = _input1.GarageYrBlt.fillna(_input1.GarageYrBlt.median(), inplace=False)
round(_input1[missing_value_df.column_name.values].isnull().sum() / len(_input1), 3)
_input1 = _input1.fillna(method='ffill', inplace=False)
_input1 = _input1.fillna(method='bfill', inplace=False)

def plot_value_counts(columns, df):
    for column in columns:
        if len(df[column].value_counts()) >= 6 and len(df[column].value_counts()) <= 15:
            plt.figure(figsize=(10, 8))
            df[column].value_counts().plot(kind='barh', fontsize=12, color='gold')
            plt.title(f'The Frequency of the {column} column', fontsize=15)
        elif len(df[column].value_counts()) < 6:
            plt.figure(figsize=(10, 8))
            df[column].value_counts().plot(kind='pie', autopct='%1.1f%%', fontsize=12)
            plt.title(f'The ratio between vlaues for the {column} column', fontsize=15)
            plt.ylabel('')
year_df = _input1.sort_values(by='YrSold')
plt.figure(figsize=(15, 8))
plt.title('The Sales Time Line According To Year', fontsize=20)
sns.lineplot(data=year_df, x='YrSold', y='SalePrice', ci=0)
plt.xlabel('The Year')
plt.figure(figsize=(15, 8))
plt.title('The Sales Time Line According To Month', fontsize=20)
sns.lineplot(data=year_df, x='MoSold', y='SalePrice', ci=0)
plt.xlabel('The Month')
from sklearn.preprocessing import LabelEncoder
for column in _input1.columns:
    if _input1[column].dtype == 'O':
        le = LabelEncoder()
        _input1[column] = le.fit_transform(_input1[column])
from sklearn.preprocessing import MinMaxScaler
col = _input1.columns
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(_input1[col])
scaled_train = pd.DataFrame(scaled_train, columns=col)
scaled_train['SalePrice'] = _input1['SalePrice']
from sklearn.model_selection import train_test_split
X = _input1.drop('SalePrice', axis=1).values
y = _input1.SalePrice.values
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
model = XGBRegressor(objective='reg:linear', max_deepth=15, seed=100, n_estimators=100, bosster='gblinear')