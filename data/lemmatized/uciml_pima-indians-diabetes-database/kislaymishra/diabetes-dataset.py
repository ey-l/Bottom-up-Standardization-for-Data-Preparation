import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
des = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
des
des.describe()
m = des[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin']].replace(0, np.nan)
m.fillna(des.mean(), inplace=True)
m
m.isnull().sum()
m1 = des['BMI']
m1
m2 = des['DiabetesPedigreeFunction']
m2
m3 = des['Age']
m3
m4 = des['Outcome']
m4
ds = pd.concat([m, m1, m2, m3, m4], axis=1)
ds
ds.isnull().sum()
ds.describe()
pass
ds.corr()
pass
pass
pass
ds['Glucose'].hist(bins=20)
ds['BloodPressure'].hist()
x = ds.drop(['Outcome'], axis=1)
x
y = ds['Outcome']
y
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(x, y, test_size=0.3, random_state=101)
X_train.shape
X_test.shape
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()