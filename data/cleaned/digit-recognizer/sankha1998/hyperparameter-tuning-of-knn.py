import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('data/input/digit-recognizer/train.csv')
test = pd.read_csv('data/input/digit-recognizer/test.csv')
sample = pd.read_csv('data/input/digit-recognizer/sample_submission.csv')
sample.head()
train.head()
X = train.iloc[:, 1:].values
y = train.iloc[:, 0]

def ch(X, y, i):
    print(y[i])
    plt.imshow(X[i].reshape(28, 28))
    X[i].reshape(1, 784)

ch(X, y, 305)
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.1)
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
param_dict = {'max_depth': [5, 10, 15, 20], 'splitter': ['best', 'random']}
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(clf, param_grid=param_dict, cv=10)