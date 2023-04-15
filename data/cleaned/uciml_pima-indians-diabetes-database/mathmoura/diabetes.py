import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
t = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
t.head().T
t.columns
t.shape
t.Outcome.value_counts()
t.describe()
sns.pairplot(t, hue='Outcome')
data_train = t.sample(frac=0.8, random_state=1)
data_test = t.drop(data_train.index)
X_train = data_train.drop(['Outcome'], axis=1)
y_train = data_train['Outcome']
X_test = data_test.drop(['Outcome'], axis=1)
y_test = data_test['Outcome']
fig = sns.FacetGrid(t, hue='Outcome', aspect=3)
fig.map(sns.kdeplot, 'Age', shade=True)
fig.add_legend()
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix
dtc = tree.DecisionTreeClassifier()