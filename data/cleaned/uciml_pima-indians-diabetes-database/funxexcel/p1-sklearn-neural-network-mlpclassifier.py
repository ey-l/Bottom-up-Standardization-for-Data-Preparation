import pandas as pd
import numpy as numpy
from sklearn.neural_network import MLPClassifier
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.info()
y = data['Outcome']
X = data.drop(['Outcome'], axis=1)
X.head()
X.shape
X.isnull().sum()
nnModel = MLPClassifier(solver='lbfgs', alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1, max_iter=1000)