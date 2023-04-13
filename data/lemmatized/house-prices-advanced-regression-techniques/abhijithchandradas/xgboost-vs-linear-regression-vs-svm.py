import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style='darkgrid')
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from warnings import filterwarnings
filterwarnings('ignore')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
print('Train shape:', _input1.shape)
print('Test Shape:', _input0.shape)
X_trainfull = _input1.drop(['SalePrice'], axis=1)
y = _input1.SalePrice
plt.figure(figsize=(8, 4))
plt.title('Distribution of Sales Price (y)')
sns.distplot(y)
y = np.log1p(y)
plt.figure(figsize=(8, 4))
plt.title('Distribution of log Sales Price (y)')
sns.distplot(y)
plt.xlabel('Log of Sales Price')
d_temp = X_trainfull.isna().sum().sort_values(ascending=False)
d_temp = d_temp[d_temp > 0]
d_temp = d_temp / _input1.shape[0] * 100
plt.figure(figsize=(8, 5))
plt.title('Features Vs Percentage Of Null Values')
sns.barplot(y=d_temp.index, x=d_temp, orient='h')
plt.xlim(0, 100)
plt.xlabel('Null Values (%)')
na_index = d_temp[d_temp > 20].index
X_trainfull = X_trainfull.drop(na_index, axis=1, inplace=False)
num_cols = X_trainfull.corrwith(y).abs().sort_values(ascending=False).index
X_num = X_trainfull[num_cols]
X_cat = X_trainfull.drop(num_cols, axis=1)
X_num.sample(5)
high_corr_num = X_num.corrwith(y)[X_num.corrwith(y).abs() > 0.5].index
X_num = X_num[high_corr_num]
plt.figure(figsize=(10, 6))
sns.heatmap(X_num.corr(), annot=True, cmap='coolwarm')
print('Correlation of Each feature with target')
X_num.corrwith(y)
X_num = X_num[high_corr_num]
X_num = X_num.drop(['TotRmsAbvGrd', 'GarageArea', '1stFlrSF', 'GarageYrBlt'], axis=1, inplace=False)

def handle_na(df, func):
    """
    Input dataframe and function 
    Returns dataframe after filling NA values
    eg: df=handle_na(df, 'mean')
    """
    na_cols = df.columns[df.isna().sum() > 0]
    for col in na_cols:
        if func == 'mean':
            df[col] = df[col].fillna(df[col].mean())
        if func == 'mode':
            df[col] = df[col].fillna(df[col].mode()[0])
    return df
X_num = handle_na(X_num, 'mean')

def scale_df(df):
    """
    Input: data frame
    Output: Returns minmax scaled Dataframe 
    eg: df=scale_df(df)
    """
    scaler = MinMaxScaler()
    for col in df.columns:
        df[col] = scaler.fit_transform(np.array(df[col]).reshape(-1, 1))
    return df
X_num = scale_df(X_num)
(X_train, X_val, y_train, y_val) = train_test_split(X_num, y, test_size=0.2)
model = LinearRegression()