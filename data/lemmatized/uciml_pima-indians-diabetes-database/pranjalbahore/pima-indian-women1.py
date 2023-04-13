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
pass
pass
pass
pass
pass
pass
pass
train.corr()['Outcome']
pass
pass
pass
train.shape
train['Glucose'] = train['Glucose'].replace(0, train['Glucose'].median())
pass
pass
train['Glucose'] = np.log(train.Glucose)
train.BloodPressure.nunique()
pass
pass
train['BloodPressure'].median()
train['BloodPressure'] = train['BloodPressure'].replace(0, train['BloodPressure'].median())
pass
pass
train['SkinThickness'].value_counts()
pass
train['SkinThickness'] = train['SkinThickness'].replace([0, 99, 7, 8], 32)
pass
train.corr()['Outcome']
pass
pass
pass
train['Insulin'].quantile(0.99)
train[train.Insulin != 0].shape
train[train.Insulin != 0]['Insulin'].mean()
train['Insulin'] = train['Insulin'].replace(0, 152.85)
pass
train['Insulin'].isnull().sum()
train.corr()['Insulin']
train.groupby(by='Glucose')['Insulin'].median()
'\ndef fill(Glucose,Insulin):\n    if pd.isnull(Insulin):\n        return Insulin == gg[Glucose]\n    else:\n        return Insulin\n'
train.Insulin = train['Insulin'].replace(0)
train.head()
train.BMI.value_counts()
pass
train['BMI'] = train['BMI'].replace(0, train.BMI.mean())
pass
pass
train.DiabetesPedigreeFunction.value_counts()
pass
train[train.DiabetesPedigreeFunction > 1].shape
pass
pass
train['DiabetesPedigreeFunction'] = np.log(train.DiabetesPedigreeFunction)
pass
train.hist(figsize=(20, 20))
train.head()
train.shape
x = train.drop('Outcome', axis=1)
y = train.Outcome
from sklearn.preprocessing import StandardScaler