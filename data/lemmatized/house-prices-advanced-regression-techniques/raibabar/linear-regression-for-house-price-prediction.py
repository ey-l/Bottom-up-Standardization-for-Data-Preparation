import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import missingno as msno
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, plot_roc_curve, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from matplotlib.dates import DateFormatter
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
obj_col = ['object']
object_columns = list(_input1.select_dtypes(include=obj_col).columns)
df_categorical = _input1[object_columns]
print()
print('There are ' + str(df_categorical.shape[1]) + ' categorical columns within dataframe:')
my_list = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
num_columns = list(_input1.select_dtypes(include=my_list).columns)
numerical_columns = _input1[num_columns]
numerical_columns
msno.bar(numerical_columns)
numerical_columns.isnull().sum()
_input1 = _input1.drop(['LotFrontage'], axis=1)
_input0 = _input0.drop(['LotFrontage'], axis=1)
_input1 = _input1.drop(['GarageYrBlt'], axis=1)
_input0 = _input0.drop(['GarageYrBlt'], axis=1)
df_categorical

def outliers_z_score(data):
    threshold = 3
    mean = np.mean(data)
    std = np.std(data)
    z_scores = [(y - mean) / std for y in data]
    return np.where(np.abs(z_scores) > threshold)
outlier_list = numerical_columns.apply(lambda x: outliers_z_score(x))
outlier_list
df_of_outlier = outlier_list.iloc[0]
df_of_outlier = pd.DataFrame(df_of_outlier)
df_of_outlier.columns = ['Rows_to_exclude']
df_of_outlier
outlier_list_final = df_of_outlier['Rows_to_exclude'].to_numpy()
outlier_list_final = np.concatenate(outlier_list_final, axis=0)
outlier_list_final_unique = set(outlier_list_final)
outlier_list_final_unique
filter_rows_to_exclude = _input1.index.isin(outlier_list_final_unique)
df_without_outliers = _input1[~filter_rows_to_exclude]
print('Length of original dataframe: ' + str(len(_input1)))
print('Length of new dataframe without outliers: ' + str(len(df_without_outliers)))
print('----------------------------------------------------------------------------------------------------')
print('Difference between new and old dataframe: ' + str(len(_input1) - len(df_without_outliers)))
print('----------------------------------------------------------------------------------------------------')
print('Length of unique outlier list: ' + str(len(outlier_list_final_unique)))
df_without_outliers = df_without_outliers.reset_index()
df_without_outliers = df_without_outliers.rename(columns={'index': 'old_index'})
df_without_outliers
for col in df_categorical.columns:
    if df_categorical[col].isnull().sum() > 0:
        total_null = df_categorical[col].isnull().sum()
        print('Column {} has total null {}, i.e. {} %'.format(col, total_null, round(total_null * 100 / len(df_categorical), 2)))
msno.bar(df_categorical)
train_df = df_without_outliers.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)
_input0 = _input0.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)
for col in train_df.columns:
    if train_df[col].isnull().sum() > 0:
        total_null = train_df[col].isnull().sum()
        print('Column {} has total null {}, i.e. {} %'.format(col, total_null, round(total_null * 100 / len(train_df), 2)))
_input1 = train_df.fillna(lambda x: x.mean())
_input1
_input0
_input0.columns.isnull().sum() > 0
_input1.columns.isnull().sum() > 0
low_cardinality_cols = [cname for cname in _input1.columns if _input1[cname].nunique() < 10 and _input1[cname].dtype == 'object']
numeric_cols = [cname for cname in _input1.columns if _input1[cname].dtype in ['int64', 'float64']]
my_cols = low_cardinality_cols + numeric_cols
predictors = _input1[my_cols]
low_cardinality_cols_t = [cname for cname in _input0.columns if _input0[cname].nunique() < 10 and _input0[cname].dtype == 'object']
numeric_cols_t = [cname for cname in _input0.columns if _input1[cname].dtype in ['int64', 'float64']]
my_cols_t = low_cardinality_cols_t + numeric_cols_t
predictors_t = _input1[my_cols_t]
print('Low Cardinality Columns in test data:')
print()
print(low_cardinality_cols_t)
print('---------------------------------------------------------------------------------------------------------------------')
print('Numeric Columns in test data:')
print()
print(numeric_cols_t)
_input0.isnull().sum()
_input0 = _input0.fillna(lambda x: x.mode())
_input0
one_hot_encoded_predictors_t = pd.get_dummies(predictors_t)
encoded_cat_predictors_t = one_hot_encoded_predictors_t
df_num_t = _input0._get_numeric_data()
Data_t = pd.concat([df_num_t, encoded_cat_predictors_t], axis=1)
Data_t = Data_t.loc[:, ~Data_t.columns.duplicated()]
X_test = Data_t
X_test.isnull().sum()
X_test
X_test = X_test.fillna(0, inplace=False)
X_test
print('Low Cardinality Columns:')
print()
print(low_cardinality_cols)
print('---------------------------------------------------------------------------------------------------------------------')
print('Numeric Columns:')
print()
print(numeric_cols)
one_hot_encoded_predictors = pd.get_dummies(predictors)
encoded_cat_predictors = one_hot_encoded_predictors
df_num = _input1._get_numeric_data()
Data = pd.concat([df_num, encoded_cat_predictors], axis=1)
Data = Data.loc[:, ~Data.columns.duplicated()]
Data.head()
X = Data.drop(['SalePrice', 'old_index'], axis=1)
y = Data.SalePrice
X.loc[:, X.columns.duplicated()].any()
X.shape
from sklearn import linear_model
regr = linear_model.LinearRegression()