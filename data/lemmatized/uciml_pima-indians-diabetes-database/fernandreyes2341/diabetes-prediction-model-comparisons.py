import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv', header=0)
df.head()
df.describe()
df.isnull().sum()
import seaborn as sb
import matplotlib.pyplot as plt
pass
sb.heatmap(df.corr(), annot=True, cmap='YlGnBu')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
explan = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'DiabetesPedigreeFunction', 'Age']
X = df[explan]
y = df['Outcome']
X.head()
y.head()
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.25, random_state=0)
logreg = LogisticRegression(solver='liblinear')