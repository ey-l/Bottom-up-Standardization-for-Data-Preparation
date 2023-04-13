"""import my libraries """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')
'load training dataset'
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input2 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
_input2 = _input2.drop(['Id'], axis=1)
_input0 = pd.concat([_input0, _input2], axis=1)
df = pd.concat([_input1, _input0], axis=0)
df.head()
df.describe()
df.isnull().sum()
lis = ['MiscFeature', 'Fence', 'PoolQC', 'Alley', 'BsmtFinSF2', '3SsnPorch', 'MiscVal', 'LowQualFinSF', 'BsmtHalfBath']
df = df.drop(lis, axis=1)
df.head()
df.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1, figsize=(12, 12))
df['SalePrice'].describe()
sns.distplot(df['SalePrice'])
corr_matrix = df.corr()
corr_mat = df.drop('Id', axis=1).corr()
(f, ax) = plt.subplots(figsize=(12, 10))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr_matrix, annot=None, cmap=cmap)
df.corr()['SalePrice'].abs()
C = corr_matrix.nlargest(5, 'SalePrice')['SalePrice'].index
for i in C:
    var = i
    data = pd.concat([df['SalePrice'], df[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
N = corr_mat.nsmallest(15, 'SalePrice')['SalePrice'].index
for n in N:
    df = df.drop(n, axis=1)
cleaning = df.drop(['SalePrice'], axis=1)
SalePrice = df['SalePrice']
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric_cols = cleaning.select_dtypes(include=numerics)
numeric_cols = numeric_cols.fillna(numeric_cols.mean())
numeric_cols.head()
categorical = ['object']
categorical_cols = cleaning.select_dtypes(include=categorical)
categorical_cols = categorical_cols.fillna('none')
categorical_cols = pd.get_dummies(categorical_cols)
categorical_cols.head()
cleaned = pd.concat([numeric_cols, categorical_cols], axis=1)
df = pd.concat([cleaned, SalePrice], axis=1)
tst_df = df.iloc[1460:, :-1]
X = df.iloc[:1460, :-1]
y = df.iloc[:1460, -1]
scl = Normalizer()
X = scl.fit_transform(X)
tst_df = scl.fit_transform(tst_df)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=4)
g = GradientBoostingRegressor(n_estimators=170, learning_rate=0.4, max_depth=2)