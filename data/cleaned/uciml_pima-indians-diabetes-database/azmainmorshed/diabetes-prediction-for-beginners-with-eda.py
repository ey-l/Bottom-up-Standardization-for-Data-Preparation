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
plt.figure(figsize=(8, 8))
sns.heatmap(corr, cbar=True, square=True, fmt='.1f', annot=True, cmap='Reds')
df['Result'].value_counts()
plt.figure(figsize=(10, 10))
sns.countplot(x='Result', data=df)
x = []
for age in df.Age:
    x.append(age)
y = df.Insulin
plt.figure(figsize=(20, 10))
plt.bar(x, y)
plt.xlabel('Age', size=10)
plt.ylabel('Insulin')
plt.xticks(x)
plt.title('Relationship between Age and Insulin levels')


x = []
for age in df.Age:
    x.append(age)
y = df.Glucose
plt.figure(figsize=(20, 10))
plt.bar(x, y)
plt.xlabel('Age', size=10)
plt.ylabel('Glucose')
plt.xticks(x)
plt.grid()
plt.title('Relationship between Age and Glucose levels')

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