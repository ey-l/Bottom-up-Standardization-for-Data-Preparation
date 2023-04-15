import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
pd.set_option('display.max_columns', train.shape[1])
train.head()
df4 = train.pivot_table(index='Outcome', values=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction'], aggfunc=np.mean)
df4
train.shape
train.describe()
train.isnull().sum()
train.info()
print(train['Outcome'].value_counts())
print('\n')
plt.figure(figsize=(8, 6))
plt.pie(train['Outcome'].value_counts(), labels=['NOT DIABETIC', 'DIABETIC'], autopct='%0.1f%%', explode=[0.05, 0])

plt.figure(figsize=(10, 8))
sns.countplot(y='Pregnancies', data=train, orient='h', color='skyblue', linewidth=2.5, edgecolor='black', order=train.Pregnancies.value_counts().index)
plt.ylabel('Pregnancies', fontweight='bold', fontsize=20)
plt.xlabel('Counts', fontweight='bold', fontsize=20)
plt.title('Countplot', fontweight='bold', fontsize=15)
train.corr()['Outcome']
plt.figure(figsize=(15, 8))
sns.heatmap(train.corr(), annot=True, cmap='BrBG')
sns.distplot(train['Glucose'])
train.shape
train['Glucose'] = train['Glucose'].replace(0, train['Glucose'].median())
sns.distplot(train['Glucose'])
sns.distplot(np.log(train['Glucose']))
train['Glucose'] = np.log(train.Glucose)
train.BloodPressure.nunique()
sns.distplot(train['BloodPressure'])
sns.distplot(train[train.BloodPressure != 0]['BloodPressure'])
train['BloodPressure'].median()
train['BloodPressure'] = train['BloodPressure'].replace(0, train['BloodPressure'].median())
sns.distplot(train['BloodPressure'])
sns.distplot(train[train.SkinThickness != 0]['SkinThickness'])
train['SkinThickness'].value_counts()
sns.distplot(train['SkinThickness'])
train['SkinThickness'] = train['SkinThickness'].replace([0, 99, 7, 8], 32)
sns.distplot(train['SkinThickness'])
train.corr()['Outcome']
plt.figure(figsize=(15, 8))
sns.heatmap(train.corr(), annot=True)
sns.distplot(train.Insulin)
train['Insulin'].quantile(0.99)
train[train.Insulin != 0].shape
train[train.Insulin != 0]['Insulin'].mean()
train['Insulin'] = train['Insulin'].replace(0, 152.85)
sns.distplot(train['Insulin'])
train['Insulin'].isnull().sum()
train.corr()['Insulin']
train.groupby(by='Glucose')['Insulin'].median()
'\ndef fill(Glucose,Insulin):\n    if pd.isnull(Insulin):\n        return Insulin == gg[Glucose]\n    else:\n        return Insulin\n'
train.Insulin = train['Insulin'].replace(0)
train.head()
train.BMI.value_counts()
sns.distplot(train.BMI)
train['BMI'] = train['BMI'].replace(0, train.BMI.mean())
sns.distplot(train.BMI)
sns.distplot(np.log(train.BMI))
train.DiabetesPedigreeFunction.value_counts()
sns.distplot(train.DiabetesPedigreeFunction)
train[train.DiabetesPedigreeFunction > 1].shape
sns.distplot(train.DiabetesPedigreeFunction)
sns.distplot(np.log(train.DiabetesPedigreeFunction))
train['DiabetesPedigreeFunction'] = np.log(train.DiabetesPedigreeFunction)
sns.distplot(train.Age)
train.hist(figsize=(20, 20))

train.head()
train.shape
x = train.drop('Outcome', axis=1)
y = train.Outcome
from sklearn.preprocessing import StandardScaler