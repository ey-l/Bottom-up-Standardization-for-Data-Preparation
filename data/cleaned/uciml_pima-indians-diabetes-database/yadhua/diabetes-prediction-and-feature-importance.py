import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
Diabetes = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
Diabetes.head()
Diabetes.info()
plt.figure(figsize=(15, 10))
sns.heatmap(Diabetes.corr(), annot=True)
plt.title('Heatmap of Variable Correlations', fontsize=25)

plt.figure(figsize=(15, 10))
sns.pairplot(Diabetes)

plt.figure(figsize=(10, 5))
sns.barplot(x='Outcome', y='Pregnancies', data=Diabetes)
plt.title('Outcome vs Pregnancies')
plt.xlabel('Outcome')
plt.ylabel('Pregnancies')

plt.figure(figsize=(10, 5))
sns.barplot(x='Outcome', y='Glucose', data=Diabetes)
plt.title('Outcome vs Glucose')
plt.xlabel('Outcome')
plt.ylabel('Glucose')

plt.figure(figsize=(10, 5))
sns.barplot(x='Outcome', y='BloodPressure', data=Diabetes)
plt.title('Outcome vs Blood Pressure')
plt.xlabel('Outcome')
plt.ylabel('Blood Pressure')

plt.figure(figsize=(10, 5))
sns.barplot(x='Outcome', y='SkinThickness', data=Diabetes)
plt.title('Outcome vs Thickness of Skin')
plt.xlabel('Outcome')
plt.ylabel('Thickness of Skin')

plt.figure(figsize=(10, 5))
sns.barplot(x='Outcome', y='Insulin', data=Diabetes)
plt.title('Outcome vs Insulin')
plt.xlabel('Outcome')
plt.ylabel('Insulin')

plt.figure(figsize=(10, 5))
sns.barplot(x='Outcome', y='BMI', data=Diabetes)
plt.title('Outcome vs Body Mass Index')
plt.xlabel('Outcome')
plt.ylabel('Body Mass Index')

plt.figure(figsize=(10, 5))
sns.barplot(x='Outcome', y='Age', data=Diabetes)
plt.title('Outcome vs Age')
plt.xlabel('Outcome')
plt.ylabel('Age')

Diabetes.head()
X = Diabetes.drop('Outcome', axis=1)
y = Diabetes['Outcome']
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size=0.7, random_state=100)
X_train.head()
y_train.head()
import statsmodels.api as sm
X_train_sm = sm.add_constant(X_train)
model = sm.GLM(y_train, X_train_sm, family=sm.families.Binomial())