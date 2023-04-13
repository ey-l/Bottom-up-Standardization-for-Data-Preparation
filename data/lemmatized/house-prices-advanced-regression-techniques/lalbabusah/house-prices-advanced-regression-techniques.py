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
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input2 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
print(_input1.shape)
_input1.head()
print(_input0.shape)
_input0.head()
print(_input2.shape)
_input2.head()
_input1.describe()

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
    df = df.sort_values(['train_isnull'], axis=0, ascending=False, inplace=False)
    return df
data_analysis(_input1, _input0).head(50)
col_drop = []
for col in _input0.columns:
    if _input1[col].isnull().sum() > 45 or _input0[col].isnull().sum() > 45:
        col_drop.append(col)
_input1 = _input1.drop(col_drop, axis=1, inplace=False)
_input0 = _input0.drop(col_drop, axis=1, inplace=False)
for col in _input0.columns:
    if _input1[col].unique().shape[0] <= 30:
        _input1[col] = _input1[col].fillna(_input1[col].mode()[0], inplace=False)
        _input0[col] = _input0[col].fillna(_input0[col].mode()[0], inplace=False)
        _input1[col] = LabelEncoder().fit_transform(_input1[col])
        _input0[col] = LabelEncoder().fit_transform(_input0[col])
for col in _input0.columns:
    if _input1[col].isnull().sum() > 0 or _input0[col].isnull().sum() > 0:
        _input1[col] = _input1[col].fillna(_input1[col].median(), inplace=False)
        _input0[col] = _input0[col].fillna(_input0[col].median(), inplace=False)
_input1.head()
_input0.head()
_input1.corr()
for col in _input1.columns:
    (pearson_coef, p_value) = stats.pearsonr(_input1[col], _input1['SalePrice'])
    print(col, pearson_coef, p_value)
col_continues = []
for col in _input0.columns:
    if _input1[col].unique().shape[0] >= _input1.shape[0] // 10:
        col_continues.append(col)
print(col_continues)
col_skew = _input1[col_continues].apply(lambda x: stats.skew(x)).sort_values(ascending=False)
high_skew = col_skew[abs(col_skew) > 0.5]
high_skew
'for col in high_skew.index:\n    df_train[col] = np.log1p(df_train[col])\n    df_test[col] = np.log1p(df_test[col])'
sns.distplot(_input1['LotArea'])
q = _input1['LotArea'].quantile(0.99)
_input1 = _input1[_input1['LotArea'] < q]
_input1 = _input1.reset_index(drop=True, inplace=False)
sns.distplot(_input1['BsmtFinSF1'])
sns.distplot(_input1['BsmtUnfSF'])
sns.distplot(_input1['TotalBsmtSF'])
sns.distplot(_input1['1stFlrSF'])
sns.distplot(_input1['GrLivArea'])
Y_train = _input1['SalePrice'].values
X_train = _input1.drop(['Id', 'SalePrice'], axis=1)
X_test = _input0.drop(['Id'], axis=1)
print(X_train.shape, Y_train.shape)
print(X_test.shape)