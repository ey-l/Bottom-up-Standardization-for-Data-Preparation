import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
dataset = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
dataset.head()
import seaborn as sns
import matplotlib.pyplot as plt
print('no of people with no diabestes ', dataset.Outcome.value_counts()[0])
print('no of people with diabestes ', dataset.Outcome.value_counts()[1])
pass
pass
heat = dataset.corr()
pass
pass
pass
pass
pass
pass
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
x = dataset.drop(['Outcome'], axis=1)
x.head()
x = np.array(x)
print(x.shape)
y = dataset[['Outcome']]
y.head()
y = np.array(y).ravel()
print(y.shape)
test = SelectKBest(score_func=f_classif, k='all')