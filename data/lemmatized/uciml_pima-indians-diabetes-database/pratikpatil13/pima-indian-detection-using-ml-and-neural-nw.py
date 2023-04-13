import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.corr()
df.isnull().sum()
df['Pregnancies'].value_counts()
df['Outcome'].value_counts()
pass
pass
df.nunique()
X = df.iloc[:, :-1].values
X
Y = df.iloc[:, -1].values
from sklearn.preprocessing import StandardScaler
one = StandardScaler()
X = one.fit_transform(X)
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(X, Y, test_size=0.1, random_state=42)
from sklearn.svm import SVC
svc = SVC(kernel='rbf')