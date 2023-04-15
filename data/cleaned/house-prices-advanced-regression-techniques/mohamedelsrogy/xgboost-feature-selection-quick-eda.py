import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
train_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train_df.head()
train_df.drop('Id', axis=1, inplace=True)
percent_missing = train_df.isnull().sum() * 100 / len(train_df)
missing_value_df = pd.DataFrame({'column_name': train_df.columns, 'percent_missing': percent_missing})
missing_value_df = missing_value_df[missing_value_df.percent_missing > 0]
plt.figure(figsize=(15, 10))
plt.barh(missing_value_df['column_name'], missing_value_df['percent_missing'], color='darkblue')
plt.title('The Percentages Of The Columns Null Values', fontsize=15)

train_df.duplicated().sum()
train_df.drop(['Fence', 'MiscFeature', 'PoolQC', 'FireplaceQu', 'Alley'], axis=1, inplace=True)
train_df.shape
percent_missing = train_df.isnull().sum() * 100 / len(train_df)
missing_value_df = pd.DataFrame({'column_name': train_df.columns, 'percent_missing': percent_missing})
missing_value_df = missing_value_df[missing_value_df.percent_missing > 0]
plt.figure(figsize=(10, 8))
train_df.boxplot(['LotFrontage', 'MasVnrArea', 'GarageYrBlt'])
plt.title('The Box Plot For the Floating Null Columns', fontsize=15)

train_df[missing_value_df.column_name.values].describe()
train_df.MasVnrArea.fillna(method='ffill', inplace=True)
train_df.MasVnrArea.fillna(method='bfill', inplace=True)
train_df.LotFrontage.fillna(train_df.LotFrontage.median(), inplace=True)
train_df.GarageYrBlt.fillna(train_df.GarageYrBlt.median(), inplace=True)
round(train_df[missing_value_df.column_name.values].isnull().sum() / len(train_df), 3)
train_df.fillna(method='ffill', inplace=True)
train_df.fillna(method='bfill', inplace=True)

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

year_df = train_df.sort_values(by='YrSold')
plt.figure(figsize=(15, 8))
plt.title('The Sales Time Line According To Year', fontsize=20)
sns.lineplot(data=year_df, x='YrSold', y='SalePrice', ci=0)
plt.xlabel('The Year')

plt.figure(figsize=(15, 8))
plt.title('The Sales Time Line According To Month', fontsize=20)
sns.lineplot(data=year_df, x='MoSold', y='SalePrice', ci=0)
plt.xlabel('The Month')

from sklearn.preprocessing import LabelEncoder
for column in train_df.columns:
    if train_df[column].dtype == 'O':
        le = LabelEncoder()
        train_df[column] = le.fit_transform(train_df[column])
from sklearn.preprocessing import MinMaxScaler
col = train_df.columns
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_df[col])
scaled_train = pd.DataFrame(scaled_train, columns=col)
scaled_train['SalePrice'] = train_df['SalePrice']
from sklearn.model_selection import train_test_split
X = train_df.drop('SalePrice', axis=1).values
y = train_df.SalePrice.values
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
model = XGBRegressor(objective='reg:linear', max_deepth=15, seed=100, n_estimators=100, bosster='gblinear')