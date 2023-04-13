import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.shape
data.sample(10)
data.info()
data.describe()
data.isnull().sum()
num_cols = []
cat_cols = []
for i in data.columns:
    if len(data[i].unique()) <= 10:
        cat_cols.append(i)
    else:
        num_cols.append(i)
num_cols
cat_cols
import matplotlib.pyplot as plt
import seaborn as sns
pass
pass
pass
for (i, feature) in enumerate(data.columns):
    pass
    data[data['Outcome'] == 0][feature].hist(bins=30, color='b', label='Have Diabetes = No', alpha=0.6)
    data[data['Outcome'] == 1][feature].hist(bins=30, color='r', label='Have Diabetes = Yes', alpha=0.6)
    pass
    pass
pass
for i in data.columns:
    if i != 'Outcome':
        pass
pass
pass
x = data.drop(['Outcome'], axis=1)
y = data['Outcome']
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.3, random_state=42)
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
bclf = BaggingClassifier(base_estimator=tree, n_estimators=50, random_state=42)