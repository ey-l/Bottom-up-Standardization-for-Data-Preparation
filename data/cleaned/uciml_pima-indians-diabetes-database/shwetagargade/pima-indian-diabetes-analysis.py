import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('ggplot')
import numpy as np
f = open('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
f.readline()
data = np.loadtxt(f, delimiter=',')
X = data[:, :-1]
y = data[:, -1]
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.25, random_state=0)
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
names = ['Nearest Neighbors', 'Linear SVM', 'RBF SVM', 'Gaussian Process', 'Decision Tree', 'Random Forest', 'Neural Net', 'AdaBoost', 'Naive Bayes', 'QDA']
classifiers = [KNeighborsClassifier(), SVC(kernel='linear'), SVC(kernel='rbf'), GaussianProcessClassifier(), DecisionTreeClassifier(), RandomForestClassifier(), MLPClassifier(), AdaBoostClassifier(), GaussianNB(), QuadraticDiscriminantAnalysis()]
from sklearn.model_selection import cross_val_score
results = {}
for (name, clf) in zip(names, classifiers):
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    results[name] = scores
for (name, scores) in results.items():
    print('%20s | Accuracy: %0.2f%% (+/- %0.2f%%)' % (name, 100 * scores.mean(), 100 * scores.std() * 2))
clf = SVC(kernel='linear', C=0.1)