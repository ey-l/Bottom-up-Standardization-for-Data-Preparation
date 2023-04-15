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
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df_train
corr_y = df_train.corr()
corr_y['SalePrice'].sort_values(ascending=False).abs()[1:]
num_columns = [col for col in df_train.columns if (df_train[col].dtype == 'int64' or df_train[col].dtype == 'float64') and col != 'Id']
corr = df_train[num_columns].corr().abs()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
plt.figure(figsize=(18, 8))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', cbar_kws={'shrink': 0.8}, vmin=0, vmax=1)

df_col = df_train[num_columns]
for i in range(0, len(num_columns), 5):
    sns.pairplot(data=df_col, x_vars=df_col.columns[i:i + 5], y_vars=['SalePrice'])
cat_columns = [col for col in df_train.columns if df_train[col].dtype == 'object']
train = df_train[num_columns]
train.fillna(train.mean(), inplace=True)
x_train = preprocessing.scale(train.iloc[:, :-1])
y_train = np.log1p(train.iloc[:, -1:])
sns.distplot(y_train)
lin_reg = LinearRegression()