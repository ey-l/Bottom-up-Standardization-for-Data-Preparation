import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df
y = df['Outcome']
X = df.drop(['Outcome'], axis=1)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_norm = pd.DataFrame(sc.fit_transform(X), columns=X.columns)
X_norm
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X_norm, y)
from sklearn.svm import SVC
clf = SVC()