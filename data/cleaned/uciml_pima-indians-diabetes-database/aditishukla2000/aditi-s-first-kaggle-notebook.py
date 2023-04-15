import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import GridSearchCV
import pydotplus as pdot

import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

diabetes = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
diabetes.info()
print(diabetes.shape)
print(diabetes.head(5))
corrMatrix = diabetes.corr()
sn.heatmap(corrMatrix, annot=True)

X = diabetes.loc[:, diabetes.columns != 'Outcome']
y = diabetes['Outcome']
(train_X, test_X, train_y, test_y) = train_test_split(X, y, test_size=0.2)

def draw_cm(actual, predicted):
    cm = metrics.confusion_matrix(actual, predicted, [1, 0])
    sn.heatmap(cm, annot=True, fmt='.2f', xticklabels=['Diabetes-Yes', 'Diabetes-No'], yticklabels=['Diabetes-Yes', 'Diabetes-No'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

rf_classifier = RandomForestClassifier(max_depth=10, n_estimators=10)