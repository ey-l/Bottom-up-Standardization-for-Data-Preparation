import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.tail()
df.shape
df.info()
df.describe()
df.isnull().any()
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
df.head()
df.isnull().any()
df.isnull().sum()
df['Glucose'].fillna(df['Glucose'].median(), inplace=True)
df['BloodPressure'].fillna(df['BloodPressure'].median(), inplace=True)
df['SkinThickness'].fillna(df['SkinThickness'].median(), inplace=True)
df['Insulin'].fillna(df['Insulin'].median(), inplace=True)
df['BMI'].fillna(df['BMI'].median(), inplace=True)
df.head()
df.hist(figsize=(20, 20))
import matplotlib.pyplot as plt
import seaborn as sns
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
corr = df.corr()
corr
pass
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df
from sklearn.model_selection import train_test_split
(train, test) = train_test_split(df, test_size=0.2, random_state=2, stratify=df['Outcome'])
train_X = train.drop(columns=['Outcome'])
test_X = test.drop(columns=['Outcome'])
train_Y = train['Outcome']
test_Y = test['Outcome']
train_X[['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'SkinThickness']] = train_X[['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'SkinThickness']].replace(0, np.NaN)
test_X[['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'SkinThickness']] = test_X[['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'SkinThickness']].replace(0, np.NaN)
for C in ['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'SkinThickness']:
    train_X[C].fillna(df[C].median(), inplace=True)
    test_X[C].fillna(df[C].median(), inplace=True)
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
normalized_train_X = scaler.fit_transform(train_X)
normalized_test_X = scaler.transform(test_X)
pd.DataFrame(normalized_train_X)
pd.DataFrame(normalized_test_X)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
forest_model = RandomForestClassifier(n_estimators=30, random_state=1, n_jobs=-1)