import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
import pandas as pd
import numpy as np
from skompiler import skompile
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter('ignore', category=ConvergenceWarning)
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.info()
df.describe().T
df.isnull().sum()
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
df.isnull().sum()

def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == 'O']
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != 'O']
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == 'O']
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != 'O']
    num_cols = [col for col in num_cols if col not in num_but_cat]
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return (cat_cols, cat_but_car, num_cols, num_but_cat)
(cat_cols, cat_but_car, num_cols, num_but_cat) = grab_col_names(df)

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.25)
    quartile3 = dataframe[variable].quantile(0.75)
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
    print(col, check_outlier(df, col))

def replace_with_thresholds(dataframe, variable):
    (low_limit, up_limit) = outlier_thresholds(dataframe, variable)
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit
for col in num_cols:
    replace_with_thresholds(df, col)
df.isnull().sum()
df.pivot_table(df, index=['Outcome'])
for col in df.columns:
    df.loc[(df['Outcome'] == 0) & df[col].isnull(), col] = df[df['Outcome'] == 0][col].median()
    df.loc[(df['Outcome'] == 1) & df[col].isnull(), col] = df[df['Outcome'] == 1][col].median()
df.loc[df['BMI'] < 18.5, 'NEW_BMI_CAT'] = 'Underweight'
df.loc[(df['BMI'] > 18.5) & (df['BMI'] < 25), 'NEW_BMI_CAT'] = 'Normal'
df.loc[(df['BMI'] > 25) & (df['BMI'] < 30), 'NEW_BMI_CAT'] = 'Overweight'
df.loc[(df['BMI'] > 30) & (df['BMI'] < 40), 'NEW_BMI_CAT'] = 'Obese'
df.loc[df['Glucose'] < 70, 'NEW_GLUCOSE_CAT'] = 'Low'
df.loc[(df['Glucose'] > 70) & (df['Glucose'] < 99), 'NEW_GLUCOSE_CAT'] = 'Normal'
df.loc[(df['Glucose'] > 99) & (df['Glucose'] < 126), 'NEW_GLUCOSE_CAT'] = 'Secret'
df.loc[(df['Glucose'] > 126) & (df['Glucose'] < 200), 'NEW_GLUCOSE_CAT'] = 'High'
df.loc[df['SkinThickness'] < 30, 'NEW_SKIN_THICKNESS'] = 'Normal'
df.loc[df['SkinThickness'] >= 30, 'NEW_SKIN_THICKNESS'] = 'HighFat'
df.loc[df['Pregnancies'] == 0, 'NEW_PREGNANCIES'] = 'NoPregnancy'
df.loc[(df['Pregnancies'] > 0) & (df['Pregnancies'] <= 4), 'NEW_PREGNANCIES'] = 'StdPregnancy'
df.loc[df['Pregnancies'] > 4, 'NEW_PREGNANCIES'] = 'OverPregnancy'
df.loc[(df['SkinThickness'] < 30) & (df['BloodPressure'] < 80), 'NEW_CIRCULATION_LEVEL'] = 'Normal'
df.loc[(df['SkinThickness'] >= 30) & (df['BloodPressure'] >= 80), 'NEW_CIRCULATION_LEVEL'] = 'CircularAtHighRisk'
df.loc[(df['SkinThickness'] < 30) & (df['BloodPressure'] >= 80) | (df['SkinThickness'] >= 30) & (df['BloodPressure'] < 80), 'NEW_CIRCULATION_LEVEL'] = 'CircularAtMediumRisk'
df['Pre_Age_Cat'] = df['Age'] * df['Pregnancies']
df['Ins_Glu_Cat'] = df['Glucose'] * df['Insulin']

def label_encoder(dataframe, binary_col):
    labelencoder = preprocessing.LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe
binary_cols = [col for col in df.columns if df[col].dtypes == 'O' and len(df[col].unique()) == 2]
for col in df.columns:
    label_encoder(df, col)

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
ohe_cols = [col for col in df.columns if 10 >= len(df[col].unique()) > 2]
one_hot_encoder(df, ohe_cols, drop_first=True)
y = df['Outcome']
X = df.drop(['Outcome'], axis=1)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=17)