import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures
df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df.head()
df.shape
df.info()
df.describe()
df.describe().T
df.describe().T.shape
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), cmap='RdBu')
plt.title('Correlations between Variables', size=15)

df.corr()
df.corr()['SalePrice']
df.corr()['SalePrice'] > 0.5
df.corr()['SalePrice'] < -0.5
list(df.corr()['SalePrice'][df.corr()['SalePrice'] > 0.5])
list(df.corr()['SalePrice'][df.corr()['SalePrice'] > 0.5].index)
print(list(df.corr()['SalePrice'][(df.corr()['SalePrice'] > 0.5) | (df.corr()['SalePrice'] < -0.5)]))
imp_cols = list(df.corr()['SalePrice'][(df.corr()['SalePrice'] > 0.5) | (df.corr()['SalePrice'] < -0.5)].index)
print(imp_cols)
print(len(imp_cols))
df['KitchenQual']
cat_cols = ['MSZoning', 'Utilities', 'BldgType', 'Heating', 'KitchenQual', 'SaleCondition', 'LandSlope']
imp = imp_cols + cat_cols
print(imp)
print(len(imp))
df = df[imp]
df
print('Missing values by column')
print('-' * 30)
print(df.isna())
print(df.isna().sum())
print('-' * 30)
print('Total Missing Values :', df.isna().sum().sum())
sns.pairplot(df[imp_cols])
imp_col1 = imp_cols
imp_col1.remove('SalePrice')
imp_col1
for i in imp_col1:
    sns.jointplot(x=df[i], y=df['SalePrice'], kind='kde')
x = df.drop('SalePrice', axis=1)
y = df['SalePrice']
print(x)
print(y)
x = pd.get_dummies(x, columns=cat_cols)
x
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
df_test.head()
imp_test = imp
imp_test.remove('SalePrice')
print(imp_test)
df_test_new = df_test[imp_test]
print(df_test_new)
x_test_new = df_test_new
print(x_test_new)
print(cat_cols)
x_test_new = pd.get_dummies(df_test_new, columns=cat_cols)
print(x_test_new)
missing_cols = set(x.columns) - set(x_test_new.columns)
missing_cols
for c in missing_cols:
    x_test_new[c] = 0
print(x_test_new)
scaler = StandardScaler()
x[imp_col1] = scaler.fit_transform(x[imp_col1])
x
x.shape
x_test_new[imp_col1] = scaler.transform(x_test_new[imp_col1])
x_test_new
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2, random_state=42)
x_train
x_test
y_train
y_test

def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, x, y, scoring='neg_mean_squared_error', cv=5)).mean()
    return rmse

def evaluation(y, predictions):
    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r_squared = r2_score(y, predictions)
    return (mae, mse, rmse, r_squared)
models = pd.DataFrame(columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score', 'RMSE (Cross-Validation)'])
lin_reg = LinearRegression()