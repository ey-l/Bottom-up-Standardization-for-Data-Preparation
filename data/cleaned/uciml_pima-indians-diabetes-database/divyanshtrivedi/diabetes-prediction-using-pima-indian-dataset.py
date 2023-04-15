import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.dtypes
data.isnull().sum()
plt.figure(figsize=(12, 10))
sns.heatmap(data.corr(), annot=True)
X = data.drop(columns='Outcome')
y = data['Outcome']
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
mn = [KNN(), DTC(), GaussianNB(), SVC(), LinearSVC(), RandomForestClassifier(), AdaBoostClassifier(), ExtraTreesClassifier(), BaggingClassifier(), GradientBoostingClassifier(), LogisticRegression(), LDA()]
for i in range(12):
    Model = mn[i]