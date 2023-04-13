import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
import scipy.stats as stats
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings(action='ignore')

def out_remove(col_name, df, cond, m):
    quartile1 = df[col_name].quantile(0.25)
    quartile3 = df[col_name].quantile(0.75)
    iqr = quartile3 - quartile1
    upper = quartile3 + m * iqr
    lower = quartile1 - m * iqr
    if cond == 'both':
        new_df = df[(df[col_name] < upper) & (df[col_name] > lower)]
    elif cond == 'lower':
        new_df = df[df[col_name] > lower]
    else:
        new_df = df[df[col_name] < upper]
    return new_df
diabetes_df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
diabetes_df.head()
diabetes_df.shape
diabetes_df.info()
pass
pass
percent_missing = diabetes_df.isnull().mean() * 100
pass
pass
num_cols = diabetes_df.columns
rcParams['figure.figsize'] = (5, 5)
pass
diabetes_df.describe().T
d_copy = diabetes_df.copy()
d_copy = d_copy.drop(columns=['Outcome'], axis=1)
d_copy = d_copy.replace(0, np.nan)
pass
pass
percent_missing = d_copy.isnull().mean() * 100
pass
pass
X = diabetes_df.iloc[:, :-1]
y = diabetes_df.iloc[:, -1]
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
X_train = pd.concat([X_train, y_train], axis=1)
X_test = pd.concat([X_test, y_test], axis=1)
print(X_train.shape)
print(y_train.shape)
rcParams['figure.figsize'] = (30, 15)
pass
pass
pass
pass
for i in range(4):
    pass
    pass
for i in range(4, 8):
    pass
    pass
pass
for i in range(1):
    pass
    pass
    pass
print(X_train.shape)
print(y_train.shape)
X_train = out_remove('Pregnancies', X_train, 'both', 1.5)
X_test = out_remove('Pregnancies', X_test, 'both', 1.5)
X_train
pass
for i in range(1):
    pass
    pass
    pass
X_train['Glucose'] = X_train['Glucose'].replace(0, X_train['Glucose'].mean())
X_test['Glucose'] = X_test['Glucose'].replace(0, X_test['Glucose'].mean())
X_train = out_remove('Glucose', X_train, 'both', 1.5)
X_test = out_remove('Glucose', X_test, 'both', 1.5)
pass
for i in range(1):
    pass
    pass
    pass
X_train['BloodPressure'] = X_train['BloodPressure'].replace(0, X_train['BloodPressure'].median())
X_test['BloodPressure'] = X_test['BloodPressure'].replace(0, X_test['BloodPressure'].median())
X_train = out_remove('BloodPressure', X_train, 'lower', 1.5)
X_test = out_remove('BloodPressure', X_test, 'lower', 1.5)
pass
for i in range(1):
    pass
    pass
    pass
X_train['SkinThickness'] = X_train['SkinThickness'].replace(0, X_train['SkinThickness'].mean())
X_test['SkinThickness'] = X_test['SkinThickness'].replace(0, X_test['SkinThickness'].mean())
X_train = out_remove('SkinThickness', X_train, 'both', 1.5)
X_test = out_remove('SkinThickness', X_test, 'both', 1.5)
pass
for i in range(1):
    pass
    pass
    pass
X_train['Insulin'] = X_train['Insulin'].replace(0, X_train['Insulin'].median())
X_test['Insulin'] = X_test['Insulin'].replace(0, X_test['Insulin'].median())
X_train = out_remove('Insulin', X_train, 'both', 1.5)
X_test = out_remove('Insulin', X_test, 'both', 1.5)
pass
for i in range(1):
    pass
    pass
    pass
X_train['BMI'] = X_train['BMI'].replace(0, X_train['BMI'].mean())
X_test['BMI'] = X_test['BMI'].replace(0, X_test['BMI'].mean())
pass
for i in range(1):
    pass
    pass
    pass
X_train = out_remove('DiabetesPedigreeFunction', X_train, 'both', 1.5)
X_test = out_remove('DiabetesPedigreeFunction', X_test, 'both', 1.5)
pass
for i in range(1):
    pass
    pass
    pass
rcParams['figure.figsize'] = (30, 15)
pass
pass
pass
pass
for i in range(4):
    pass
    pass
for i in range(4, 8):
    pass
    pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
from sklearn.tree import DecisionTreeClassifier
pass
model = DecisionTreeClassifier()
z_copy = X_train.copy()
y = z_copy.iloc[:, -1]
X = z_copy.iloc[:, :-1]