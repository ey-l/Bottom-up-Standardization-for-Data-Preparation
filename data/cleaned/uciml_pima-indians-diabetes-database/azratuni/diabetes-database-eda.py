import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
warnings.simplefilter('ignore')
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')


df.info()
df.describe()
df.describe().columns
df.shape
df.isnull().values.any()
df.isnull().sum()
print((df[df.columns] == 0).sum())
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
df.head()
df.isnull().sum()
fig = plt.figure(figsize=(10, 6))
y = range(768)
plt.scatter(df['BMI'], y)
plt.title('BMI')

fig = plt.figure(figsize=(10, 6))
y = range(768)
plt.scatter(df['Glucose'], y)
plt.title('Glucose')

fig = plt.figure(figsize=(10, 6))
y = range(768)
plt.scatter(df['BloodPressure'], y)
plt.title('BloodPressure')

df['Glucose'].fillna(df['Glucose'].median(), inplace=True)
df['BloodPressure'].fillna(df['BloodPressure'].median(), inplace=True)
df['BMI'].fillna(df['BMI'].median(), inplace=True)
by_Glucose_Age_Insulin_Grp = df.groupby(['Glucose'])

def fill_Insulin(series):
    return series.fillna(series.median())
df['Insulin'] = by_Glucose_Age_Insulin_Grp['Insulin'].transform(fill_Insulin)
df['Insulin'] = df['Insulin'].fillna(df['Insulin'].mean())
by_BMI_Insulin = df.groupby(['BMI'])

def fill_Skinthickness(series):
    return series.fillna(series.mean())
df['SkinThickness'] = by_BMI_Insulin['SkinThickness'].transform(fill_Skinthickness)
df['SkinThickness'].fillna(df['SkinThickness'].mean(), inplace=True)
df.isnull().sum()
df['Glucose'].value_counts()
df['Age'].value_counts()
labels = ['Healthy', 'Diabetic']
df['Outcome'].value_counts().plot(kind='pie', labels=labels, subplots=True, autopct='%1.0f%%')
y = df['Pregnancies'].value_counts()
plt.figure(figsize=(6, 6))
sns.barplot(x='Pregnancies', y=y, data=df)

Pregnancies1 = df['Pregnancies'].dropna()
sns.distplot(Pregnancies1)

Glucose1 = df['Glucose'].dropna()
sns.distplot(Glucose1)

BloodPressure1 = df['BloodPressure'].dropna()
sns.distplot(BloodPressure1)

SkinThickness1 = df['SkinThickness'].dropna()
sns.distplot(SkinThickness1)

Insulin1 = df['Insulin'].dropna()
sns.distplot(Insulin1)

BMI1 = df['BMI'].dropna()
sns.distplot(BMI1)

DiabetesPedigreeFunction1 = df['DiabetesPedigreeFunction'].dropna()
sns.distplot(DiabetesPedigreeFunction1)

Age1 = df['Age'].dropna()
sns.distplot(Age1)
