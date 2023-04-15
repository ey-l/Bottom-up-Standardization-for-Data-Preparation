import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
import lightgbm as lgb
import matplotlib.pyplot as plt
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train.head()
print(f'Training Set :\n Number of rows : {train.shape[0]}, Number of Columns : {train.shape[1]}')
print(f'Test Set :\n Number of rows : {test.shape[0]}, Number of Columns : {test.shape[1]}')
num_cols = train.loc[:, train.dtypes != 'object'].drop(['Id'], axis=1).columns
num_train = train[num_cols]
cat_cols = train.loc[:, train.dtypes == 'object'].columns
cat_train = train[cat_cols]
print('Total Numerical Cols : ', len(num_cols))
print('Total Categorical Cols : ', len(cat_cols))
train.describe()
train.info()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
sns.distplot(train['SalePrice'], fit=norm)