import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import KFold, cross_val_score
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df_train
corr_y = df_train.corr()
corr_y['SalePrice'].sort_values(ascending=False).abs()[1:]
num_columns = [col for col in df_train.columns if df_train[col].dtype == 'int64']
corr = df_train[num_columns].corr().abs()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
plt.figure(figsize=(15, 8))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', cbar_kws={'shrink': 0.8}, vmin=0, vmax=1)

num_col2 = [x for x in num_columns if x != 'Id' and x != 'MoSold']
(num_col2, len(num_col2))
train = df_train[num_col2]
x_train = preprocessing.scale(train.iloc[:, :-1])
y_train = np.log1p(train.iloc[:, -1:])
sns.distplot(y_train)
for i in range(0, len(train.columns), 5):
    sns.pairplot(data=train, x_vars=train.columns[i:i + 5], y_vars=['SalePrice'])
lin_reg = LinearRegression()
cross_val_score(lin_reg, x_train, y_train, cv=4, scoring='neg_root_mean_squared_error')
sgd_reg = SGDRegressor(max_iter=5000, tol=-np.infty, penalty=None, eta0=0.005, random_state=42)
cross_val_score(sgd_reg, x_train, y_train, cv=4, scoring='neg_root_mean_squared_error')
kf = KFold(4, shuffle=True, random_state=1)
cross_val_score(sgd_reg, x_train, y_train.to_numpy().ravel(), cv=kf, scoring='neg_root_mean_squared_error')