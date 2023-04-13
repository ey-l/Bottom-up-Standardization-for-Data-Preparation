import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
import matplotlib.pyplot as plt
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
print(df.head())
print(df.isnull().sum())
for i in df.columns:
    print(i, len(df[df[i] == 0]))
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)
print(df.isnull().sum())
df = df.fillna(method='ffill')
df = df.fillna(method='bfill')
print(df.isnull().sum())
pass
list1_col = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
for i in range(len(list1_col)):
    row = i // 2
    col = i % 2
    ax = axes1[row, col]
    pass
print(df.Insulin.shape)
print(df.SkinThickness.shape)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
df = df[~((df.iloc[:, 4:5] < Q1 - 1.5 * IQR) | (df.iloc[:, 4:5] > Q3 + 1.5 * IQR)).any(axis=1)]
df = df[~((df.iloc[:, 3:4] < Q1 - 1.5 * IQR) | (df.iloc[:, 3:4] > Q3 + 1.5 * IQR)).any(axis=1)]
print(df.Insulin.shape)
print(df.SkinThickness.shape)
pass
list1_col = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
for i in range(len(list1_col)):
    row = i // 2
    col = i % 2
    ax = axes1[row, col]
    pass
x = df.drop('Outcome', axis=1)
y = df.Outcome
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.3, random_state=1)
print(x_train.head())
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='liblinear')