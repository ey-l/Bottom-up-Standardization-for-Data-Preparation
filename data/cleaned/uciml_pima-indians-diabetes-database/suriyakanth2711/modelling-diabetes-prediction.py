import numpy as np
import pandas as pd
import missingno as msno
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.info()
df.corr()
msno.matrix(df)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
x = df.drop('Outcome', axis=1)
y = df.Outcome
(x_train, x_test, y_train, y_test) = train_test_split(x, y, random_state=1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
lr = LogisticRegression()