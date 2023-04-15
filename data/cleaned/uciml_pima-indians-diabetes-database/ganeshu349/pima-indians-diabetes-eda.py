import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.describe()
df.isna().sum()
df.duplicated().sum()
plt.figure(figsize=(12, 6))
sns.countplot(x='Outcome', data=df)

plt.figure(figsize=(12, 12))
for (i, col) in enumerate(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(data=df, x=col)

sns.pairplot(df, hue='Outcome')

plt.figure(figsize=(12, 12))
for (i, col) in enumerate(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']):
    plt.subplot(3, 3, i + 1)
    sns.histplot(x=col, data=df, kde=True)

plt.figure(figsize=(12, 12))
sns.heatmap(df.corr(), annot=True, robust=False)

import missingno as msno
msno.matrix(df)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = pd.DataFrame(sc.fit_transform(df.drop(['Outcome'], axis=1)), columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
X.head()
y = df['Outcome']
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.neighbors import KNeighborsClassifier
test_scores = []
train_scores = []
for i in range(1, 15):
    knn = KNeighborsClassifier(i)