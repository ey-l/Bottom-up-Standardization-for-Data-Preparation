import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imblearn
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.describe()
df.head(5)
df['Outcome'].value_counts().plot(kind='pie', autopct='%.1f%%', radius=1.5)
l = []
l2 = []
not_obese = 0
obese = 0
very_obese = 0
for x in df['Outcome'].index:
    if df['Outcome'][x] == 1:
        l.append(df['BMI'][x])
for j in l:
    temp = int(j)
    l2.append(temp)
for i in l2:
    if i in l2 and i in range(1, 26):
        not_obese = not_obese + 1
    elif i in l2 and i in range(25, 31):
        obese = obese + 1
    else:
        very_obese = very_obese + 1
data = [not_obese, obese, very_obese]
labels = ['not obese', 'obese', 'very obese']
pass
x = df['DiabetesPedigreeFunction']
y = df['Outcome']
pass
nonzero_mean_gl = df['Glucose'][df['Glucose'] != 0].mean()
df['Glucose'] = df['Glucose'].replace(0, nonzero_mean_gl, inplace=False)
nonzero_mean_bp = df['BloodPressure'][df['BloodPressure'] != 0].mean()
df['BloodPressure'] = df['BloodPressure'].replace(0, nonzero_mean_bp, inplace=False)
nonzero_mean_st = df['SkinThickness'][df['SkinThickness'] != 0].mean()
df['SkinThickness'] = df['SkinThickness'].replace(0, nonzero_mean_st)
nonzero_mean_in = df['Insulin'][df['Insulin'] != 0].mean()
df['Insulin'] = df['Insulin'].replace(0, nonzero_mean_in)
nonzero_mean_bmi = df['BMI'][df['BMI'] != 0].mean()
df['BMI'] = df['BMI'].replace(0, nonzero_mean_bmi)
df.describe()
classifier = svm.SVC(kernel='linear', C=100, gamma='auto', degree=3)
X = df.iloc[:, :-1]
y = df.iloc[:, -1:]
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline = make_pipeline(classifier)