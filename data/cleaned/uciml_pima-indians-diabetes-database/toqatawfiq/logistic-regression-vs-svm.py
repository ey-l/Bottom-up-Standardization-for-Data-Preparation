import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
X_full = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
X_full.shape
X_full.info()
X_full.describe()
print('Number of rows missing Glucose: {0}'.format(len(X_full.loc[X_full['Glucose'] == 0])))
print('Number of rows missing Blood Pressure: {0}'.format(len(X_full.loc[X_full['BloodPressure'] == 0])))
print('Number of rows missing Insulin: {0}'.format(len(X_full.loc[X_full['Insulin'] == 0])))
print('Number of rows missing BMI: {0}'.format(len(X_full.loc[X_full['BMI'] == 0])))
print('Number of rows missing Skin Thickness: {0}'.format(len(X_full.loc[X_full['SkinThickness'] == 0])))
print('Number of rows missing Age: {0}'.format(len(X_full.loc[X_full['Age'] == 0])))
x = X_full['Glucose'].median()
X_full['Glucose'].replace(0, x, inplace=True)
x = X_full['BloodPressure'].median()
X_full['BloodPressure'].replace(0, x, inplace=True)
x = X_full['Insulin'].median()
X_full['Insulin'].replace(0, x, inplace=True)
x = X_full['BMI'].median()
X_full['BMI'].replace(0, x, inplace=True)
x = X_full['SkinThickness'].median()
X_full['SkinThickness'].replace(0, x, inplace=True)
figure = plt.figure(figsize=(10, 10))
sns.heatmap(X_full.corr(), annot=True, cmap='YlGnBu')
X_full['Outcome'].value_counts()
X_full.isna().sum(axis=0)
len(X_full['Age'].unique())
figure = plt.figure(figsize=(6, 6))
ax = sns.histplot(x='Age', hue='Outcome', data=X_full, multiple='dodge', shrink=0.8)
plt.hist(X_full['BMI'], bins=100)

X_full['BMI_levels'] = pd.cut(X_full['BMI'], bins=[0, 18.5, 24.99, 29.99, 34.99, 39.99, 100], labels=[0, 1, 2, 3, 4, 5])
plt.hist(X_full['BMI_levels'])

figure = plt.figure(figsize=(6, 6))
ax = sns.histplot(x='BMI_levels', hue='Outcome', data=X_full, multiple='dodge', shrink=0.8)
plt.hist(X_full['Glucose'], bins=100)

X_full['Glucose_levels'] = pd.cut(X_full['Glucose'], bins=[0, 140, 200], labels=[0, 1])
figure = plt.figure(figsize=(6, 6))
ax = sns.histplot(x='Glucose_levels', hue='Outcome', data=X_full, multiple='dodge', shrink=0.8)
X_full.head()
X_org = X_full.drop(columns=['BMI_levels', 'Glucose_levels'])
Y = X_full['Outcome']
X = X_full.drop(columns=['Outcome'])
(X_train, X_test, y_train, y_test) = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
from sklearn import svm