import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
pd.set_option('display.max_columns', None)
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.describe().T
zero_columns = [i for i in df.columns if df[i].min() == 0 and i not in ['Pregnancies', 'Outcome']]
for i in zero_columns:
    df[[i]] = df[[i]].replace(0, np.NaN)
df.isnull().sum()
for i in zero_columns:
    df[i] = df[i].fillna(df.groupby('Outcome')[i].transform('median'))
df.isnull().values.any()
num_cols = [col for col in df.columns if col != 'Outcome']

def outlier_thresholds(dataframe, col_name):
    quartile1 = dataframe[col_name].quantile(0.1)
    quartile3 = dataframe[col_name].quantile(0.9)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return (low_limit, up_limit)

def check_outlier(dataframe, col_name):
    (low_limit, up_limit) = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
for col in num_cols:
    print(col, ':', check_outlier(df, col))

def replace_with_thresholds(dataframe, col_name):
    (low_limit, up_limit) = outlier_thresholds(dataframe, col_name)
    if low_limit > 0:
        dataframe.loc[dataframe[col_name] < low_limit, col_name] = low_limit
        dataframe.loc[dataframe[col_name] > up_limit, col_name] = up_limit
    else:
        dataframe.loc[dataframe[col_name] > up_limit, col_name] = up_limit
replace_with_thresholds(df, 'Insulin')
replace_with_thresholds(df, 'SkinThickness')
replace_with_thresholds(df, 'DiabetesPedigreeFunction')
for col in num_cols:
    print(col, ':', check_outlier(df, col))
for i in num_cols:
    pass
    pass
    pass
for col in num_cols:
    print(df.groupby('Outcome').agg({col: ['min', 'mean', 'median', 'std', 'max']}))
    print('-' * 30)
for col in num_cols:
    for i in df['Outcome'].unique():
        (test_stast, pvalue) = stats.shapiro(df.loc[df['Outcome'] == i, col])
        print(f'Numeric variable: {col} Class: {i} pvalue: {pvalue}')
for col in num_cols:
    (test_stats, pvalue) = stats.mannwhitneyu(df.loc[df['Outcome'] == df['Outcome'].unique()[0], col], df.loc[df['Outcome'] == df['Outcome'].unique()[1], col])
    print(f'Numeric variable: {col} pvalue: {pvalue}')
for col in num_cols:
    print(df.groupby('Outcome').agg({col: 'median'}))
    print('-' * 30)
df['PREG_AGE'] = df['Pregnancies'] * df['Age']
df['Glucose_BMI'] = df['Glucose'] * df['BMI']
df['Insulin_Glucose'] = df['Insulin'] * df['Glucose']
df['Insulin_BMI'] = df['Insulin'] * df['BMI']
df['INSULİN_AGE'] = df['Insulin'] * df['Age']

def set_insulin(row):
    if row['Insulin'] >= 16 and row['Insulin'] <= 166:
        return 'Normal'
    else:
        return 'Abnormal'
df = df.assign(NewInsulinScore=df.apply(set_insulin, axis=1))
df.head()
binary_cols = [col for col in df.columns if df[col].dtypes == 'O' and df[col].nunique() == 2]

def label_encoder(dataframe, binary_col):
    labelencoder = preprocessing.LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe
for col in binary_cols:
    df = label_encoder(df, col)
df.head()
y = df['Outcome']
x = df.drop('Outcome', axis=1)
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.3, random_state=42)
print(f'{x_train.shape}, {x_test.shape}, {y_train.shape}, {y_test.shape}')
cls = DecisionTreeClassifier()