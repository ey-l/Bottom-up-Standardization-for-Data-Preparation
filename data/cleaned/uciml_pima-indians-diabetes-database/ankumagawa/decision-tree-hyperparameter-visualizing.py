import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df
df.describe()
df.info()
df['Outcome'].value_counts()
X = df.iloc[:, :8]
y = df['Outcome']
X
y
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=3)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
clf = DecisionTreeClassifier()