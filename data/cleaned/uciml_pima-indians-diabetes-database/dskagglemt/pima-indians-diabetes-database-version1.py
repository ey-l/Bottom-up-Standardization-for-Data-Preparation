import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv', names=col_names)
pima.head()
pima = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv', skiprows=1, names=col_names)
pima.head()
pima.shape
pima.columns
feature_cols = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age']
X = pima[feature_cols]
y = pima.label
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.25, random_state=0)

lr = LogisticRegression()