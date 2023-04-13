import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import seaborn as sns
import matplotlib.pyplot as plt
df_data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df_data.T
df_data.describe().T
pass
pass
a = df_data.hist(figsize=(20, 20))
data = df_data.copy()
for attr in ['Glucose', 'BloodPressure', 'Insulin', 'BMI']:
    data[attr] = data[attr].replace(0, data[attr].mean())
a = data.hist(figsize=(20, 20))
X = data.to_numpy()[:, :-1]
Y = data.to_numpy()[:, -1]
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, Y, test_size=0.3)
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
for kernel in ('linear', 'sigmoid', 'rbf'):
    clf = svm.SVC(kernel=kernel)