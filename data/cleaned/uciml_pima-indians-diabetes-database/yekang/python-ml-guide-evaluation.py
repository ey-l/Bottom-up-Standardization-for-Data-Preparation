import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Binarizer
diabetes_data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
print(diabetes_data['Outcome'].value_counts())
diabetes_data.head()
diabetes_data.info()
X = diabetes_data.iloc[:, :-1]
y = diabetes_data.iloc[:, -1]
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=156, stratify=y)

def get_clf_eval(y_test, pred):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred)
    print('오차 행렬')
    print(confusion)
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f},    F1: {3:.4f}, AUC:{4:.4f}'.format(accuracy, precision, recall, f1, roc_auc))
lr_clf = LogisticRegression()