import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df = df.rename(columns={'Outcome': 'Result'})
df.head()
df.info()
corr = df.corr()
pass
pass
df['Result'].value_counts()
pass
pass
x = []
for age in df.Age:
    x.append(age)
y = df.Insulin
pass
pass
pass
pass
pass
pass
x = []
for age in df.Age:
    x.append(age)
y = df.Glucose
pass
pass
pass
pass
pass
pass
pass
df.groupby('Result').mean()
X = df.drop(['Result'], axis=1)
y = df['Result']
Scaler = StandardScaler()
StandardizedData = Scaler.fit_transform(X)
print(StandardizedData)
X = StandardizedData
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=2)
X.shape
X_train.shape
model = svm.SVC(kernel='linear')