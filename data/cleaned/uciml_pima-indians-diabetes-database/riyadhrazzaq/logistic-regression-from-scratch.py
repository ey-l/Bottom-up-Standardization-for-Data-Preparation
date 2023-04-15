import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X = (X - np.min(X)) / (np.max(X) - np.min(X))
(X_train, X_test, y_train, y_test) = train_test_split(X, y)

def pipeline(X, y, X_test, y_test, alpha, max_iter, bs):
    """
    Sklearn Sanity Check
    """
    print('-' * 20, 'Sklearn', '-' * 20)
    sk = LogisticRegression(penalty='none', max_iter=max_iter)