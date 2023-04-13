import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')
pd.pandas.set_option('display.max_columns', None)
pd.pandas.set_option('display.max_rows', None)
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1.shape
_input1.head()
_input1.describe().T
_input1.info()
cols_na = _input1.columns[_input1.isnull().any()]
print('Missing Values by Column')
print('-' * 30)
for col in cols_na:
    print(col, _input1[col].count(), _input1[col].isnull().sum(), np.round(_input1[col].isnull().mean() * 100, 2), '%')
sns.pairplot(_input1[cols_na])
num_cols_na = [col for col in _input1.columns[_input1.isnull().any()] if _input1[col].dtype != 'O']
for col in num_cols_na:
    plt.figure(figsize=(10, 8))
    sns.jointplot(x=_input1[col], y=_input1['SalePrice'], kind='kde')
df_na = _input1.copy()
cat_cols_na = [col for col in df_na.columns[df_na.isnull().any()] if df_na[col].dtype == 'O']
df_na = df_na[cat_cols_na + ['SalePrice']]
for col in cat_cols_na:
    df_na[col] = np.where(df_na[col].isnull(), 1, 0)
    df_na.groupby(col)['SalePrice'].mean().plot.bar()
    plt.xlabel(col)
    plt.ylabel('Mean House Price')
    plt.title(col)
df_na.head()
num_cols = _input1.select_dtypes(exclude='object').columns
_input1[num_cols].shape
_input1[num_cols].head()
for col in num_cols:
    print(col, _input1[col].dtype)
temp_cols = [col for col in num_cols if 'Yr' in col or 'Year' in col]
for col in temp_cols:
    print(col, _input1[col].dtype)
for col in temp_cols:
    _input1.groupby(col)['SalePrice'].mean().plot()
    plt.xlabel(col)
    plt.ylabel('Mean House Price')
    plt.title('Mean House Price vs. ' + col)
df_temp = _input1.copy()
for col in temp_cols[:-1]:
    df_temp[col] = df_temp['YrSold'] - df_temp[col]
    plt.scatter(df_temp[col], df_temp['SalePrice'])
    plt.xlabel('House age since ' + col)
    plt.ylabel('Sale Price')
    plt.title('House Price vs. house age since ' + col)
dis_cols = [col for col in num_cols if len(_input1[col].unique()) < 25 and col not in temp_cols]
con_cols = [col for col in num_cols if col not in dis_cols and col not in temp_cols + ['Id']]
_input1[dis_cols].head()
_input1[con_cols].head()
for col in dis_cols:
    _input1.groupby(col)['SalePrice'].mean().plot.bar()
    plt.xlabel(col)
    plt.ylabel('Mean House Price')
    plt.title('Mean House Price vs. ' + col)
for col in con_cols:
    _input1[col].hist(bins=25)
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.title(col)
for col in con_cols:
    df_con = _input1.copy()
    if 0 in df_con[col].unique() or col in ['SalePrice']:
        pass
    else:
        df_con[col] = np.log(df_con[col])
        df_con['SalePrice'] = np.log(df_con['SalePrice'])
        plt.scatter(df_con[col], df_con['SalePrice'])
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.title(col)
cat_cols = _input1.select_dtypes(include='object').columns
_input1[cat_cols].head()
for col in cat_cols:
    print(col, len(_input1[col].unique()))
for col in cat_cols:
    _input1.groupby(col)['SalePrice'].mean().plot.bar()
    plt.xlabel(col)
    plt.ylabel('Mean House Price')
    plt.title('Mean House Price vs. ' + col)
for col in con_cols:
    df_con = _input1.copy()
    if 0 in df_con[col].unique():
        pass
    else:
        df_con[col] = np.log(df_con[col])
        df_con.boxplot(column=col)
        plt.ylabel(col)
        plt.title(col)
df2 = _input1.copy()

def replaceNull(df, colList, type):
    if type == 'categorical':
        new_df = df.copy()
        new_df[colList] = new_df[colList].fillna('n/a')
        return new_df
    if type == 'numerical':
        new_df = df.copy()
        for col in colList:
            new_df[col] = new_df[col].fillna(new_df[col].median())
        return new_df
cat_na = [col for col in df2[cat_cols] if df2[col].isnull().sum() > 0]
for col in cat_na:
    print(col, np.round(df2[col].isnull().mean() * 100, 2), '% missing values')
df2 = df2.drop(columns={'Alley', 'PoolQC', 'Fence', 'MiscFeature'})
new_cat_cols = df2.select_dtypes(include='object').columns
new_cat_cols
df3 = replaceNull(df2, new_cat_cols, 'categorical')
for col in new_cat_cols:
    print(col, df3[col].isnull().sum() > 0)
num_na = [col for col in df3[num_cols] if df3[col].isnull().sum() > 0]
for col in num_na:
    print(col, np.round(df3[col].isnull().mean() * 100, 2), '% missing values')
df4 = replaceNull(df3, num_na, 'numerical')
for col in num_na:
    print(col, df4[col].isnull().sum() > 0)
df5 = df4.copy()
for col in temp_cols:
    if col != 'YrSold':
        df5[col] = df5['YrSold'] - df5[col]
df5[temp_cols].head()
col_num = [col for col in df5.columns if col not in ['Id', 'SalePrice'] and df5[col].dtype != 'O']

def diagnostic_plt(df, col):
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    df[col].hist()
    plt.subplot(1, 2, 2)
    stats.probplot(df[col], dist='norm', plot=plt)
    plt.title(col)
for col in col_num:
    diagnostic_plt(df5, col)
df6 = df5.copy()
skewed_cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'MiscVal', 'SalePrice']
for col in skewed_cols:
    df6[col] = np.log(df6[col] + 1)
df6.head()
cat_cols = df5.select_dtypes(include='object').columns
df5[cat_cols].head()
df7 = df6.copy()
for col in cat_cols:
    temp = df7.groupby(col)['SalePrice'].count() / len(df7)
    temp_df = temp[temp > 0.01].index
    df7[col] = np.where(df7[col].isin(temp_df), df7[col], 'rare_val')
for col in cat_cols:
    temp = np.round(df7.groupby(col)['SalePrice'].count() / len(df7) * 100, 2)
    print(temp)
df8 = df7.copy()
for col in cat_cols:
    label_encoder = df8[col].value_counts().index
    label_encoder = {j: i for (i, j) in enumerate(label_encoder)}
    df8[col] = df8[col].map(label_encoder)
cat_cols = df8.select_dtypes(include='object').columns
print(cat_cols)
df8.head()
df9 = df8.copy()
scale_cols = [col for col in df9.columns if col not in ['Id', 'SalePrice']]
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))