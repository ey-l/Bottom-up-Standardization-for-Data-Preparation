import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
raw_data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
raw_data.head()
temp = raw_data.drop(labels=['Pregnancies', 'Outcome'], axis=1)
for i in temp.columns:
    print((temp[i] == 0).sum().sum())
for i in temp.columns:
    temp[i] = temp[i].replace(0, round(temp[i].mean(), 2))
temp.head()
raw_data[temp.columns] = temp[temp.columns]
data = raw_data
data.head()
data.describe()
(fig, ax) = plt.subplots(2, 3, figsize=(15, 7))
sns.distplot(data['Glucose'], ax=ax[0][0])
sns.distplot(data['BloodPressure'], ax=ax[0][1])
sns.distplot(data['SkinThickness'], ax=ax[0][2])
sns.distplot(data['Insulin'], ax=ax[1][0])
sns.distplot(data['BMI'], ax=ax[1][1])
sns.distplot(data['DiabetesPedigreeFunction'], ax=ax[1][2])
initial_data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
a = round(initial_data['Insulin'].mean(), 2)
b = round(initial_data['SkinThickness'].mean(), 2)
(a, b)
data = data.drop(raw_data[(raw_data.Insulin == a) & (raw_data.SkinThickness == b)].index)
sns.distplot(data['SkinThickness'])
sns.distplot(data['Insulin'])
data.head()
targets = data['Outcome']
inputs = data.drop('Outcome', axis=1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()