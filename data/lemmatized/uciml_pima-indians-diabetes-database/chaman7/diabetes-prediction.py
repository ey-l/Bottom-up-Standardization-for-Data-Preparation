import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.columns
df.head()
df.describe()
df.shape
df.isnull().values.any()
df.hist(figsize=(20, 20))
df.groupby('Outcome').size()
pass
df.plot(kind='box', figsize=(20, 10))
df = df[df['SkinThickness'] < 80]
df = df[df['Insulin'] <= 600]
df.shape
corrmat = df.corr()
pass
pass
df.corr()
print('total number of rows : {0}'.format(len(df)))
print('number of missing pregnancies: {0}'.format(len(df.loc[df['Pregnancies'] == 0])))
print('number of missing glucose: {0}'.format(len(df.loc[df['Glucose'] == 0])))
print('number of missing bp: {0}'.format(len(df.loc[df['BloodPressure'] == 0])))
print('number of missing skinthickness: {0}'.format(len(df.loc[df['SkinThickness'] == 0])))
print('number of missing insulin: {0}'.format(len(df.loc[df['Insulin'] == 0])))
print('number of missing bmi: {0}'.format(len(df.loc[df['BMI'] == 0])))
print('number of missing diabetespedigree: {0}'.format(len(df.loc[df['DiabetesPedigreeFunction'] == 0])))
print('number of missing age: {0}'.format(len(df.loc[df['Age'] == 0])))
df.loc[df['Insulin'] == 0, 'Insulin'] = df['Insulin'].mean()
df.loc[df['Glucose'] == 0, 'Glucose'] = df['Glucose'].mean()
df.loc[df['BMI'] == 0, 'BMI'] = df['BMI'].mean()
df.loc[df['BloodPressure'] == 0, 'BloodPressure'] = df['BloodPressure'].mean()
df.loc[df['SkinThickness'] == 0, 'SkinThickness'] = df['SkinThickness'].mean()
df.head()
pass
df.hist(figsize=(20, 20))
df = df / df.max()
df.head()
X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]
X.head(10)
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
l = []
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
classifier = SVC(kernel='linear', random_state=42)