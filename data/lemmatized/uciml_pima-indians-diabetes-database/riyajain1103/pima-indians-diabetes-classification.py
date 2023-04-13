import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.columns
df.isnull().sum()
X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
X.head()
y = df['Outcome']
y.head()
X.info()
X.shape
corr = df.corr()
pass
pass
pass
X['Pregnancies'].max()
X['Pregnancies'].min()
X['Pregnancies'].mode()
X['Pregnancies'].mean()
a = df['Pregnancies'] == 17
b = df[a]
b.shape
b.head()
c = df['Pregnancies'] == df['Pregnancies'].min()
d = df[c]
d.head()
pass
pass
c = df['Glucose'] == df['Glucose'].min()
d = df[c]
d.head()
c = df['Glucose'] == df['Glucose'].max()
d = df[c]
d.head()
c = df['Glucose'] == df['Glucose'].median()
d = df[c]
d.head()
df['Glucose'].mean()
c = df['Glucose'] > df['Glucose'].mean()
d = df[c]
d.head(700)
d['Outcome'].mode()
d['Outcome'].value_counts()
c = df['Glucose'] < df['Glucose'].mean()
d = df[c]
d.head(700)
d['Outcome'].mode()
d['Outcome'].count()
d['Outcome'].value_counts()
pass
pass
pass
df['BloodPressure'].mean()
df['BloodPressure'].max()
df['BloodPressure'].min()
df['BloodPressure'].mode()
df['BloodPressure'].value_counts()
c = df['BloodPressure'] < df['BloodPressure'].mean()
d = df[c]
d.head(700)
d['Outcome'].mode()
d['Outcome'].count()
d['Outcome'].value_counts()
c = df['BloodPressure'] > df['BloodPressure'].mean()
d = df[c]
d.head(700)
d['Outcome'].mode()
d['Outcome'].count()
d['Outcome'].value_counts()
pass
pass
df['SkinThickness'].mean()
df['SkinThickness'].max()
df['SkinThickness'].min()
pass
c = df['SkinThickness'] < df['SkinThickness'].mean()
d = df[c]
d.head(700)
d['SkinThickness'].value_counts()
d['Outcome'].mode()
d['Outcome'].count()
d['Outcome'].value_counts()
105 / (105 + 246)
c = df['SkinThickness'] > df['SkinThickness'].mean()
d = df[c]
d.head(700)
d['SkinThickness'].value_counts()
d['Outcome'].value_counts()
163 / (163 + 254)
pass
pass
df.plot(x='SkinThickness', y='BloodPressure', style='o')
pass
df['Insulin'].describe()
c = df['Insulin'] == 846
d = df[c]
d.head()
c = df['Insulin'] == 0
d = df[c]
d.head()
pass
pass
c = df['Insulin'] > df['Insulin'].mean()
d = df[c]
d.head(700)
d['Insulin'].value_counts()
d['Outcome'].value_counts()
121 / (121 + 168)
c = df['Insulin'] < df['Insulin'].mean()
d = df[c]
d.head(700)
d['Insulin'].value_counts()
d['Outcome'].value_counts()
147 / (147 + 332)
pass
df.plot(x='Insulin', y='BloodPressure', style='o')
pass
pass
pass
df['BMI'].describe()
c = df['BMI'] < df['BMI'].mean()
d = df[c]
d.head(700)
d['Outcome'].value_counts()
84 / (84 + 289)
d['BMI'].value_counts()
c = df['BMI'] > df['BMI'].mean()
d = df[c]
d.head(700)
d['Outcome'].value_counts()
184 / (184 + 211)
pass
pass
pass
pass
pass
pass
pass
pass
pass
df['DiabetesPedigreeFunction'].describe()
pass
pass
c = df['DiabetesPedigreeFunction'] < df['DiabetesPedigreeFunction'].mean()
d = df[c]
d.head(700)
d['Outcome'].value_counts()
139 / (139 + 334)
c = df['DiabetesPedigreeFunction'] > df['DiabetesPedigreeFunction'].mean()
d = df[c]
d.head(700)
d['Outcome'].value_counts()
129 / (129 + 166)
pass
pass
pass
df['Age'].describe()
pass
pass
pass
pass
from sklearn import preprocessing
X = preprocessing.normalize(X, norm='l2')
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=0)
X
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)