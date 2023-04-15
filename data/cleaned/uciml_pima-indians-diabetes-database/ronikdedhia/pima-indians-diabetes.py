import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import seaborn as sns
import matplotlib.pyplot as plt

diabetes = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
diabetes.head()
diabetes.groupby('Outcome').size()
sns.countplot(diabetes['Outcome'], label='Count')
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(diabetes.loc[:, diabetes.columns != 'Outcome'], diabetes['Outcome'], stratify=diabetes['Outcome'], random_state=66)
from sklearn.neighbors import KNeighborsClassifier
training_accuracy = []
test_accuracy = []
neighbors_settings = range(1, 11)
for n_neighbors in neighbors_settings:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)