import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pass
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
pass
pass
pass
pass
pass
pass
pass
initial_data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
a = round(initial_data['Insulin'].mean(), 2)
b = round(initial_data['SkinThickness'].mean(), 2)
(a, b)
data = data.drop(raw_data[(raw_data.Insulin == a) & (raw_data.SkinThickness == b)].index)
pass
pass
data.head()
targets = data['Outcome']
inputs = data.drop('Outcome', axis=1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()