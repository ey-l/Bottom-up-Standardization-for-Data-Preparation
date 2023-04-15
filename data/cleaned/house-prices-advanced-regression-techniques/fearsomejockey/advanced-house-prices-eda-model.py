import numpy as np
import pandas as pd
import pandas_profiling
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
import xgboost
dataset = '_data/input/house-prices-advanced-regression-techniques/train.csv'
data = pd.read_csv(dataset)
data = data.drop('Id', axis=1)
data.head()
data.shape
s = pd.Series(data.isnull().sum())
print(s)
null_Columns = []
for i in range(len(s)):
    if s[i] > 0:
        null_Columns.append(i)
null_Columns
numerical_list = list(data.select_dtypes(exclude=['object']).columns.tolist())
categorical_list = list(data.select_dtypes(include=['object']).columns.tolist())
for i in numerical_list:
    data[i] = data[i].fillna(data[i].median())
for i in categorical_list:
    data[i] = data[i].fillna(data[i].mode().iloc[0])
data.isnull().sum()
y = data['SalePrice']
data = data.drop('SalePrice', axis=1)
X = pd.get_dummies(data)
stdscaler = StandardScaler()
X = stdscaler.fit_transform(X)
pca = PCA(0.98)
X_PCA = pca.fit_transform(X)
X_PCA.shape
(X_train, X_test, y_train, y_test) = train_test_split(X_PCA, y, test_size=0.2)
random_forest = RandomForestRegressor(n_estimators=1000, random_state=100)