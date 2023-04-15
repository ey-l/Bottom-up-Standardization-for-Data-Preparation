import matplotlib.pyplot as plt
import seaborn as srn
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.linear_model import *
from sklearn.preprocessing import *
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import *
from sklearn.neighbors import *
from sklearn import svm
from sklearn.naive_bayes import *
import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df
df.info()
ar1 = []
ar2 = []
ar3 = []
ar4 = []
ar5 = []
ar6 = []
colm = []
for col in df.columns:
    if col != 'Outcome':
        colm.append(col)
        x1 = df[df['Outcome'] == 0]
        x2 = df[df['Outcome'] == 1]
        m1 = max(df[col])
        m2 = min(df[col])
        ar1.append((np.mean(x1[col]) - m2) / (m1 - m2))
        ar2.append((np.mean(x2[col]) - m2) / (m1 - m2))
        ar3.append((np.median(x1[col]) - m2) / (m1 - m2))
        ar4.append((np.median(x2[col]) - m2) / (m1 - m2))
        ar5.append((x1[col].mode() - m2) / (m1 - m2))
        ar6.append((x2[col].mode() - m2) / (m1 - m2))
plt.plot(colm, ar1, label='mean output : 0')
plt.legend()
plt.plot(colm, ar2, label='mean output : 1')
plt.legend()
plt.xticks(rotation=300)

plt.plot(colm, ar3, label='median output : 0')
plt.legend()
plt.plot(colm, ar4, label='median output : 1')
plt.legend()
plt.xticks(rotation=300)

plt.plot(colm, ar5, label='mode output : 0')
plt.legend()
plt.plot(colm, ar6, label='mode output : 1')
plt.legend()
plt.xticks(rotation=300)

df.duplicated().value_counts()
model = RandomForestRegressor()
X = df.drop('Outcome', 1)
y = df['Outcome']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2)