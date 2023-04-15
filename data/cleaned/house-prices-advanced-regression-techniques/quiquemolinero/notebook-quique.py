import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV
path_data = '_data/input/house-prices-advanced-regression-techniques/'
df_train = pd.read_csv(path_data + 'train.csv')
df_test = pd.read_csv(path_data + 'test.csv')
df_test.head(5)
df_train.columns
df_train.head(5)
df_train.shape
df_train['SalePrice'].describe()
print('Menores que la media: ', df_train[df_train['SalePrice'] <= df_train['SalePrice'].mean()]['Id'].count())
print('Mayores que la media: ', df_train[df_train['SalePrice'] > df_train['SalePrice'].mean()]['Id'].count())
plt.figure(figsize=(12, 10))
plt.hist(df_train['SalePrice'], bins=25)
df_nulls = df_train.isnull().sum().sort_values(ascending=False)
df_nulls
df_ratio_nulls = (df_nulls / len(df_train)).reset_index()
df_ratio_nulls.columns = ['Feature', 'ratio_nulls']
df_ratio_nulls[df_ratio_nulls['ratio_nulls'] > 0]
cat_cols = ['MSSubClass', 'YrSold', 'MoSold']
for col in cat_cols:
    df_train[col] = df_train[col].astype(str)
df_train[cat_cols]
num_cols = ['GarageArea', 'GarageCars', 'MasVnrArea']
for col in num_cols:
    df_train[col] = df_train[col].fillna(0)
df_train.columns
df_train['OverallQual'].unique()
df_train[df_train['GarageCars'] == 4]
df_train['MiscVal'].unique()
df_train['SalePrice'][df_train['MiscVal'] == df_train['MiscVal'].unique().max()]
df_train['TotRmsAbvGrd'].describe()
df_train['Id'][df_train['TotRmsAbvGrd'].isna() == 'True'].count()
df_train['YearBuilt'].describe()
df_train['Id'][df_train['YearBuilt'].isna() == 'True'].count()
df_train['Fireplaces'].describe()
df_train['Id'][df_train['Fireplaces'] == 3].count()
df_train['Id'][df_train['Fireplaces'] >= 1].count()
df_train['Id'][df_train['Fireplaces'].isna() == 'True'].count()
df_train['Id'][df_train['KitchenQual'].isna() == 'True'].count()
df_train['KitchenQual']
from sklearn import preprocessing
lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(df_train['KitchenQual'])
df_train['KitchenQual'] = encoded.astype('int')
df_train['KitchenQual'].unique()
lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(df_train['ExterQual'])
df_train['ExterQual'] = encoded.astype('int')
lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(df_train['HeatingQC'])
df_train['HeatingQC'] = encoded.astype('int')
df_train['HeatingQC'].unique()
encoded = lab_enc.fit_transform(df_test['KitchenQual'])
df_test['KitchenQual'] = encoded.astype('int')
encoded = lab_enc.fit_transform(df_test['ExterQual'])
df_test['ExterQual'] = encoded.astype('int')
encoded = lab_enc.fit_transform(df_test['HeatingQC'])
df_test['HeatingQC'] = encoded.astype('int')
list_selected_vars = ['GarageArea', 'GarageCars', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BsmtUnfSF', 'TotalBsmtSF', 'OverallQual', 'GrLivArea', 'TotRmsAbvGrd', 'YearBuilt', 'Fireplaces', 'KitchenQual', 'HeatingQC', 'ExterQual']
list_target_col = ['SalePrice']
df_train_complete = df_train.copy()
df_train = df_train[list_selected_vars + list_target_col]
for var in list_selected_vars:
    if len(df_train[var].unique()) <= 20:
        (f, ax) = plt.subplots(figsize=(12, 8))
        plt.title(f'Variable: {var}', fontsize=14)
        sns.boxplot(x=var, y='SalePrice', data=df_train)
    else:
        (f, ax) = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=var, y='SalePrice', data=df_train)
df_train
target_cols = 'SalePrice'
corr_mat = df_train.corr()
corr_mat
df_train.shape
corr_mat.shape
target_col = 'SalePrice'
corr_mat = df_train_complete.corr()
(f, ax) = plt.subplots(figsize=(18, 12))
sns.heatmap(corr_mat)
(df_train, df_val) = train_test_split(df_train, test_size=0.2, random_state=12)
print(len(df_train), len(df_val))

def get_metric_competition_error(y_true, y_pred):
    y_true = np.log1p(y_true)
    y_pred = np.log1p(y_pred)
    msle = mean_squared_error(y_true, y_pred)
    rmsle = np.sqrt(msle)
    return rmsle
sale_price_mean = df_train['SalePrice'].mean()
train_rmsle = get_metric_competition_error(df_train['SalePrice'], [sale_price_mean for _ in range(len(df_train))])
val_rmsle = get_metric_competition_error(df_val['SalePrice'], [sale_price_mean for _ in range(len(df_val))])
print(f'Train Metric: {train_rmsle}, Validation Metric: {val_rmsle}')
model_linear = LinearRegression()