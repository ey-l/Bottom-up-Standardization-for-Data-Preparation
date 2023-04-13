import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import io
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.ensemble as skens
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
df_train = _input1.drop(['Id'], axis=1)
df_test = _input0.drop(['Id'], axis=1)
sns.distplot(df_train['SalePrice']).set_title('Sale Price Distribution Training Dataset')
df_train['logSalePrice'] = np.log(df_train['SalePrice'])
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
target = df_train['SalePrice']
target_log = df_train['logSalePrice']
sns.distplot(df_train['logSalePrice']).set_title('Log Sale Price Distribution Training Dataset')
df = _input1.append(_input0, ignore_index=True, sort=False)
df.shape
percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'column_name': df.columns, 'percent_missing': percent_missing})
dropped_columns = []
for row in missing_value_df.values:
    if row[1] > 10 and row[0] != 'SalePrice':
        dropped_columns.append(row[0])
df = df.drop(dropped_columns, axis=1)
df.shape
na = df.isnull().sum()
na[na > 0]
df['MSSubClass'] = df['MSSubClass'].astype(str)
df['YrSold'] = df['YrSold'].astype(str)
df['MoSold'] = df['MoSold'].astype(str)
df.dtypes.to_frame().reset_index()[0].value_counts()
continuous_cols = []
for col in df.columns.values:
    if df[col].dtype != 'object':
        continuous_cols.append(col)
continuous_cols
df_cont = df[continuous_cols]
df_cat = df.drop(continuous_cols, axis=1)
imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
imputed_cont = imp_mean.fit_transform(df_cont)
newlst = []
for array in imputed_cont:
    newlst.append(array)
imputed_df = pd.DataFrame(newlst)
labels = df_cont.columns.tolist()
imputed_df.columns = labels
df_cont = imputed_df
from sklearn.impute import SimpleImputer
df_cat = df_cat.fillna(0, inplace=False)
imp_mean = SimpleImputer(missing_values=0, strategy='most_frequent')