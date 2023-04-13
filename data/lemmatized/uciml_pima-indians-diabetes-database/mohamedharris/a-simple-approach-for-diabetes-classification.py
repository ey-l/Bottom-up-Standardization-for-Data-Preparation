import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import warnings
warnings.filterwarnings('ignore')
pass
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
original = df.copy()
print('Data has', df.shape[0], 'rows and', df.shape[1], 'columns')
df.info()
df.head()
df.describe()
import missingno as msna
msna.matrix(df)
df['Outcome'].value_counts(normalize=True)
pass

def univariate_plot(x):
    pass
    pass
    pass
    pass
    n = len(x)
    a = np.sort(x)
    b = np.arange(1, 1 + n) / n
    pass
    pass
    mean_x = np.mean(x)
    pass
    skew = '               Skew : ' + str(round(x.skew(), 2))
    pass
    pass
    pass
univariate_plot(df['Age'])
univariate_plot(df['BMI'])
univariate_plot(df['Pregnancies'])
univariate_plot(df['Glucose'])
univariate_plot(df['BloodPressure'])
univariate_plot(df['SkinThickness'])
univariate_plot(df['Insulin'])
univariate_plot(df['DiabetesPedigreeFunction'])
df.loc[df['Pregnancies'] > 10, :]
df.loc[df['BMI'] == 0]
df.loc[df['Glucose'] == 0]
df.loc[df['BloodPressure'] == 0]
df.loc[df['SkinThickness'] == 0]
df.loc[df['Insulin'] == 0]
drop_index = df.loc[(df['BMI'] == 0) & (df['BloodPressure'] == 0) & (df['SkinThickness'] == 0) & (df['Insulin'] == 0), :].index
df.drop(drop_index, axis=0, inplace=True)
for i in df.columns.tolist():
    print(i, '-', len(df.loc[df[i] == 0, :]))
df.sample(20)
df['Outcome'] = df['Outcome'].astype('str')
pass

def plot_box(x):
    pass
    pass
    pass
    pass
    pass
    pass
plot_box('Age')
plot_box('Pregnancies')
plot_box('Insulin')
plot_box('BMI')
plot_box('BloodPressure')
plot_box('SkinThickness')
plot_box('DiabetesPedigreeFunction')
plot_box('Glucose')
df['BMI'] = np.where(df['BMI'] == 0, np.nan, df['BMI'])
df['Glucose'] = np.where(df['Glucose'] == 0, np.nan, df['Glucose'])
df['BloodPressure'] = np.where(df['BloodPressure'] == 0, np.nan, df['BloodPressure'])
df['SkinThickness'] = np.where(df['SkinThickness'] == 0, np.nan, df['SkinThickness'])
df['BMI'].fillna(27, inplace=True)
df['Glucose'].fillna(df['Glucose'].mean(), inplace=True)
df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace=True)
df['SkinThickness'].fillna(df['SkinThickness'].mean(), inplace=True)
df.describe()
df.loc[df['SkinThickness'] > 60]
df.drop(df.loc[df['SkinThickness'] > 60].index, axis=0, inplace=True)
df.loc[df['BMI'] > 55]
df.drop(df.loc[df['BMI'] > 55].index, axis=0, inplace=True)
df.describe()
for i in df.select_dtypes(['int64', 'float64']).columns.tolist():
    print(i, ':', df[i].skew())

def skew_visual():
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
skew_visual()
for i in ['Pregnancies', 'Insulin', 'Age', 'DiabetesPedigreeFunction']:
    print(i, ':', np.sqrt(df[i]).skew())
for i in ['Pregnancies', 'Insulin', 'Age', 'DiabetesPedigreeFunction']:
    print(i, ':', np.log1p(df[i]).skew())
pd.DataFrame({'Feature': ['Pregnancies', 'Insulin', 'Age', 'DiabetesPedigreeFunction'], 'Actual': [df[i].skew() for i in df[['Pregnancies', 'Insulin', 'Age', 'DiabetesPedigreeFunction']]], 'Squared': [np.sqrt(df[i]).skew() for i in df[['Pregnancies', 'Insulin', 'Age', 'DiabetesPedigreeFunction']]], 'Cubed': [(df[i] ** (1 / 3)).skew() for i in df[['Pregnancies', 'Insulin', 'Age', 'DiabetesPedigreeFunction']]], 'Logged': [np.log1p(df[i]).skew() for i in df[['Pregnancies', 'Insulin', 'Age', 'DiabetesPedigreeFunction']]]})
df_v1 = df.copy()
df['Pregnancies_trans'] = np.sqrt(df['Pregnancies'])
df['Insulin_trans'] = np.log1p(df['Insulin'])
df['Age_trans'] = np.log1p(df['Age'])
df['DiabetesPedigreeFunction_trans'] = df['DiabetesPedigreeFunction'] ** (1 / 3)

def skew_visual_trans():
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
skew_visual_trans()
df['Pregnancies_bin'] = np.where(df['Pregnancies'] == 0, 0, np.where((df['Pregnancies'] > 0) & (df['Pregnancies'] <= 5), 1, np.where((df['Pregnancies'] > 5) & (df['Pregnancies'] <= 10), 2, 3)))
df['Insulin_bin'] = np.where(df['Insulin'] == 0, 0, np.where((df['Insulin'] > 0) & (df['Insulin'] <= 50), 1, np.where((df['Insulin'] > 50) & (df['Insulin'] <= 200), 2, 3)))
bins = np.arange(0, 100, 10)
names = [1, 2, 3, 4, 5, 6, 7, 8, 9]
df['Age_bin'] = pd.cut(df['Age'], bins=bins, labels=names)
df.sample(10)
df_v2 = df.copy()
df = df[['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'DiabetesPedigreeFunction_trans', 'Age_bin', 'Insulin_bin', 'Pregnancies_bin', 'Outcome']]
df.sample(10)
df['Outcome'] = df['Outcome'].astype('int64')
df_scaled_mms = df.copy()
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
cols = df_scaled_mms.columns.tolist()
df_scaled_mms = pd.DataFrame(mms.fit_transform(df), columns=cols)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df_scaled_sc = pd.DataFrame(sc.fit_transform(df), columns=cols)
df_scaled_mms.describe()
df_scaled_sc.describe()
pass
pass
pass
pass
pass
pass
pass
df_scaled_mms.head()
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
X = df_scaled_mms.drop(columns=['Insulin_bin', 'Outcome'])
Y = df_scaled_mms['Outcome']
(train_x, test_x, train_y, test_y) = train_test_split(X, Y, test_size=0.25, stratify=Y, random_state=42)
print('Train_X - ', train_x.shape)
print('Test_X - ', test_x.shape)
print('Train_Y - ', train_y.shape)
print('Test_Y - ', test_y.shape)

def model_build(x):
    model = x