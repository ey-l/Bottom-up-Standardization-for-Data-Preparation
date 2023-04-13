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
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
print(f'Training Set :\n Number of rows : {_input1.shape[0]}, Number of Columns : {_input1.shape[1]}')
print(f'Test Set :\n Number of rows : {_input0.shape[0]}, Number of Columns : {_input0.shape[1]}')
num_cols = _input1.loc[:, _input1.dtypes != 'object'].drop(['Id'], axis=1).columns
num_train = _input1[num_cols]
cat_cols = _input1.loc[:, _input1.dtypes == 'object'].columns
cat_train = _input1[cat_cols]
print('Total Numerical Cols : ', len(num_cols))
print('Total Categorical Cols : ', len(cat_cols))
_input1.describe()
_input1.info()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
sns.distplot(_input1['SalePrice'], fit=norm)