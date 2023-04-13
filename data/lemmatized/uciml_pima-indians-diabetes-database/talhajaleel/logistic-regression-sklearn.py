import pandas as pd
import numpy as np
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv', header=None, names=col_names)
pima = pima.iloc[1:, :]
pima.head()
feature_cols = ['pregnant', 'insulin', 'bmi', 'age', 'glucose', 'bp', 'pedigree']
X = pima[feature_cols]
y = pima.label
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.25, random_state=0)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()