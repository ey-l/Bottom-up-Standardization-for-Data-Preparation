import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, auc, accuracy_score
from sklearn import tree
from sklearn import ensemble
t = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
t.head().T
t.columns
t.count()
t.shape
X = t.drop(['Outcome'], axis=1)
y = t.Outcome
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=1)
print(X_train.shape)
print(X_test.shape)
lr = LogisticRegression(solver='liblinear')