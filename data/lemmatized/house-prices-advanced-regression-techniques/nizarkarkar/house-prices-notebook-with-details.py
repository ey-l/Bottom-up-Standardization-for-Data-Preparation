import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import ensemble
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from matplotlib.legend_handler import HandlerLine2D
from sklearn.preprocessing import LabelEncoder
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
(_input1.shape, _input0.shape)
_input1.head()
_input0.head()
_input1.dtypes
idn = _input0['Id']
_input1.isnull().sum()
_input1.info()
_input1.keys()
_input1['SalePrice'].hist(bins=100)
print('the skewness of the target is %f ' % _input1['SalePrice'].skew())
print('the kurtosis of the target is %f' % _input1['SalePrice'].kurt())
cormat = _input1.corr()
plt.subplots(figsize=(40, 22))
sns.set(font_scale=1.45)
sns.heatmap(cormat, square=True, cmap='coolwarm')
corrolation = cormat['SalePrice'].sort_values(ascending=False)
features = corrolation.index[:10]
features
sns.pairplot(_input1[features], size=1.5)
_input1 = _input1.drop(['Id'], axis=1, inplace=False)
_input0 = _input0.drop(['Id'], axis=1, inplace=False)
train_df = _input1.copy()
test_df = _input0.copy()
train_cat = _input1.select_dtypes(include='object')
train_cat.isna().sum()
test_cat = test_df.select_dtypes(include='object')
test_cat.isna().sum()

def transform(train, cat):
    for i in cat.columns:
        _input1[i] = _input1[i].fillna('None', inplace=False)
    return _input1
df_train = transform(train_df, train_cat)
df_test = transform(test_df, test_cat)
cat = df_train.select_dtypes(include='object')
from category_encoders import CountEncoder
enc = CountEncoder(normalize=True, cols=cat.columns)
df_train = enc.fit_transform(df_train)
cat_test = df_test.select_dtypes(include='object')
enct = CountEncoder(normalize=True, cols=cat_test.columns)
df_test = enct.fit_transform(df_test)
df_test.info()
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
estimator = RandomForestRegressor(max_depth=8)
mice = IterativeImputer(estimator=estimator, random_state=11, skip_complete=True)
final_train = mice.fit_transform(df_train)
final_train = pd.DataFrame(final_train, columns=df_train.columns)
final_test = mice.fit_transform(df_test)
final_test = pd.DataFrame(final_test, columns=df_test.columns)
final_test.info()
final_train['LogPrice'] = np.log(final_train['SalePrice'])
final_train.head()
X_train = final_train.drop(['SalePrice', 'LogPrice'], axis=1)
y_train = final_train['LogPrice']
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
(X_training, X_valid, y_training, y_valid) = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
lm = LinearRegression()