import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.shape
df.info()
df.describe().T
df.isnull().any()
for x in df.columns:
    print(x, len(df[df[x] == 0]))
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)
print(df.isnull().sum())
p = df.hist(figsize=(15, 15))
df['Glucose'].fillna(df['Glucose'].mean(), inplace=True)
df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace=True)
df['SkinThickness'].fillna(df['SkinThickness'].median(), inplace=True)
df['Insulin'].fillna(df['Insulin'].mean(), inplace=True)
df['BMI'].fillna(df['BMI'].median(), inplace=True)
print(df.isnull().sum())
df.Outcome.value_counts()
plt.figure(figsize=(5, 5))
sns.countplot(data=df, x='Outcome')
plt.xticks(ticks=[0, 1], labels=["Haven't Diabetes", 'Have Diabetes'])
percent_have_diabetes = len(df[df.Outcome == 1]) / len(df) * 100
percent_havenot_diabetes = len(df[df.Outcome == 0]) / len(df) * 100
print('Percent of people that have diabetes is :', round(percent_have_diabetes, 2), '%')
print("Percent of people that haven't diabetes is :", round(percent_havenot_diabetes, 2), '%')
plt.figure(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True)
100 * df.corr()['Outcome'].sort_values()
(fig, ax) = plt.subplots(2, 4, figsize=(15, 15))
sns.histplot(df.Age, bins=20, ax=ax[0, 0], color='red', kde=True, stat='density')
sns.histplot(df.Pregnancies, bins=20, ax=ax[0, 1], color='red', kde=True, stat='density')
sns.histplot(df.Glucose, bins=20, ax=ax[0, 2], color='red', kde=True, stat='density')
sns.histplot(df.BloodPressure, bins=20, ax=ax[0, 3], color='red', kde=True, stat='density')
sns.histplot(df.SkinThickness, bins=20, ax=ax[1, 0], color='red', kde=True, stat='density')
sns.histplot(df.Insulin, bins=20, ax=ax[1, 1], color='red', kde=True, stat='density')
sns.histplot(df.DiabetesPedigreeFunction, bins=20, ax=ax[1, 2], color='red', kde=True, stat='density')
sns.histplot(df.BMI, bins=20, ax=ax[1, 3], color='red', kde=True, stat='density')
plt.figure(figsize=(10, 10))
ax = sns.boxplot(data=df, orient='h', palette='Set2')
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
from sklearn.preprocessing import PolynomialFeatures
feature = PolynomialFeatures(degree=3)
X = feature.fit_transform(X)
X.shape
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.15, random_state=42)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

def models(X_train, y_train):
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(solver='liblinear')