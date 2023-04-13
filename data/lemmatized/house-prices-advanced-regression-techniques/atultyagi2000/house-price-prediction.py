import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head(3)
_input0.head(3)
data = pd.concat((_input1, _input0)).reset_index(drop=True)
data.shape
data.describe()
plt.figure(figsize=(29, 22))
sns.heatmap(data.corr(), cmap='Blues')
data.isnull().sum().sort_values(ascending=False)
plt.figure(figsize=(19, 12))
sns.heatmap(data.isnull(), yticklabels=False)

def missing_data(data):
    total = data.isnull().sum().sort_values(ascending=False)
    percent = total / len(data) * 100
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    plt.figure(figsize=(29, 6))
    plt.xticks(rotation='90')
    sns.barplot(x=missing_data[missing_data['Percent'] > 0].index, y=missing_data[missing_data['Percent'] > 0].Percent)
missing_data(data)

def drop_columns(data):
    drop_columns = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage']
    return data.drop(drop_columns, 1)
data = drop_columns(data)
missing_data(data)

def full_nan_by_mean_mod(data):
    nan_columns = data.columns[data.isnull().any()]
    for i in nan_columns:
        if data[i].dtypes == 'object':
            data[i] = data[i].fillna(data[i].mode()[0])
        else:
            data[i] = data[i].fillna(data[i].mean())
    return data
full_nan_by_mean_mod(data)
data.isnull().sum().sort_values(ascending=False)
categorical = data.dtypes == 'object'
categorical_list = list(categorical[categorical].index)
print(categorical_list)

def encodee():
    for i in categorical_list:
        encode = LabelEncoder()
        data[i] = encode.fit_transform(data[i])
encodee()
data.head(4)
data.shape
df_train = data[:1460]
df_test = data[1460:]
print(df_train.shape)
print(df_test.shape)
X = df_train.drop(['SalePrice'], 1)
y = df_train['SalePrice']
df_test = df_test.drop(['SalePrice'], 1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
df_test = scaler.transform(df_test)
model = GradientBoostingRegressor(random_state=58, n_estimators=500, loss='huber', max_depth=3, max_features=25)