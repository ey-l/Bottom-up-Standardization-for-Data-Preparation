import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.isnull().sum()
sns.scatterplot(x=df['Outcome'], y=df['Insulin'])
sns.barplot(x=df['Outcome'], y=df['Insulin'])
sns.barplot(x=df['Outcome'], y=df['Age'])
sns.barplot(x=df['Outcome'], y=df['Glucose'])
sns.barplot(x=df['Outcome'], y=df['DiabetesPedigreeFunction'])
sns.barplot(x=df['Outcome'], y=df['BMI'])
sns.barplot(x=df['Outcome'], y=df['Glucose'])
sns.barplot(x=df['Outcome'], y=df['BloodPressure'])
X = df.drop('Outcome', axis=1)
y = df['Outcome']
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.svm import LinearSVC
svc = LinearSVC()