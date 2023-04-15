import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
from scipy import optimize, stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, log_loss, mean_squared_error

df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
df_submission = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
print(df_train.shape)
df_train.head()
print(df_test.shape)
df_test.head()
print(df_submission.shape)
df_submission.head()
df_train.describe()

def data_analysis(df1, df2):
    print(df1.shape, df2.shape)
    train_dtype = []
    train_isnull = []
    train_unique = []
    test_dtype = []
    test_isnull = []
    test_unique = []
    for col in df2.columns:
        train_dtype.append(df1[col].dtypes)
        train_isnull.append(df1[col].isnull().sum())
        train_unique.append(df1[col].unique().shape[0])
        test_dtype.append(df2[col].dtypes)
        test_isnull.append(df2[col].isnull().sum())
        test_unique.append(df2[col].unique().shape[0])
    df = pd.DataFrame({'train_dtype': train_dtype, 'test_dtype': test_dtype, 'train_isnull': train_isnull, 'test_isnull': test_isnull, 'train_unique': train_unique, 'test_unique': test_unique}, index=df2.columns)
    df.sort_values(['train_isnull'], axis=0, ascending=False, inplace=True)
    return df
data_analysis(df_train, df_test).head(50)
col_drop = []
for col in df_test.columns:
    if df_train[col].isnull().sum() > 45 or df_test[col].isnull().sum() > 45:
        col_drop.append(col)
df_train.drop(col_drop, axis=1, inplace=True)
df_test.drop(col_drop, axis=1, inplace=True)
for col in df_test.columns:
    if df_train[col].unique().shape[0] <= 30:
        df_train[col].fillna(df_train[col].mode()[0], inplace=True)
        df_test[col].fillna(df_test[col].mode()[0], inplace=True)
        df_train[col] = LabelEncoder().fit_transform(df_train[col])
        df_test[col] = LabelEncoder().fit_transform(df_test[col])
for col in df_test.columns:
    if df_train[col].isnull().sum() > 0 or df_test[col].isnull().sum() > 0:
        df_train[col].fillna(df_train[col].median(), inplace=True)
        df_test[col].fillna(df_test[col].median(), inplace=True)
df_train.head()
df_test.head()
df_train.corr()
for col in df_train.columns:
    (pearson_coef, p_value) = stats.pearsonr(df_train[col], df_train['SalePrice'])
    print(col, pearson_coef, p_value)
col_continues = []
for col in df_test.columns:
    if df_train[col].unique().shape[0] >= df_train.shape[0] // 10:
        col_continues.append(col)
print(col_continues)
col_skew = df_train[col_continues].apply(lambda x: stats.skew(x)).sort_values(ascending=False)
high_skew = col_skew[abs(col_skew) > 0.5]
high_skew
'for col in high_skew.index:\n    df_train[col] = np.log1p(df_train[col])\n    df_test[col] = np.log1p(df_test[col])'
sns.distplot(df_train['LotArea'])
q = df_train['LotArea'].quantile(0.99)
df_train = df_train[df_train['LotArea'] < q]
df_train.reset_index(drop=True, inplace=True)
sns.distplot(df_train['BsmtFinSF1'])
sns.distplot(df_train['BsmtUnfSF'])
sns.distplot(df_train['TotalBsmtSF'])
sns.distplot(df_train['1stFlrSF'])
sns.distplot(df_train['GrLivArea'])
Y_train = df_train['SalePrice'].values
X_train = df_train.drop(['Id', 'SalePrice'], axis=1)
X_test = df_test.drop(['Id'], axis=1)
print(X_train.shape, Y_train.shape)
print(X_test.shape)