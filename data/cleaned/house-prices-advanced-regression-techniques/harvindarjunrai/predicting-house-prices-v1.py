import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from scipy import stats
from scipy.stats import chi2_contingency
from scipy.stats import chi2
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
data_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
data_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print(f'Training set shape: {data_train.shape}\n')
print(f'Test set shape: {data_test.shape}\n')
data_train.info()
dif_1 = [x for x in data_train.columns if x not in data_test.columns]
print(f'Columns present in df_train and absent in df_test: {dif_1}\n')
dif_2 = [x for x in data_test.columns if x not in data_train.columns]
print(f'Columns present in df_test set and absent in df_train: {dif_2}')
data_train.drop(['Id'], axis=1, inplace=True)
Id_test_list = data_test['Id'].tolist()
data_test.drop(['Id'], axis=1, inplace=True)
print(f'Training set shape: {data_train.shape}\n')
print(f'Test set shape: {data_test.shape}\n')
data_train_num = data_train.select_dtypes(include=[np.number])
data_test_num = data_test.select_dtypes(include=[np.number])
data_train_num.head()
fig_ = data_train_num.hist(figsize=(16, 20), bins=50, color='deepskyblue', edgecolor='black', xlabelsize=8, ylabelsize=8)
sel = VarianceThreshold(threshold=0.05)