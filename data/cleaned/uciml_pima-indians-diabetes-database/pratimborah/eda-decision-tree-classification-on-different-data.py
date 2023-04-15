import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix
pima = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
pima.head()
pima.describe()
pima.info()
pima.duplicated().sum()
pima.isnull().sum()
pima.dtypes
corr = pima.corr()
sns.heatmap(corr, annot=True)
ax = sns.histplot(pima['Glucose'])

sns.pairplot(pima)
ax = sns.countplot(y='Outcome', data=pima)
ax = sns.countplot(y='Pregnancies', data=pima)
sns.lineplot(data=pima, x='Age', y='Glucose')
feature_cols = ['Pregnancies', 'Insulin', 'BMI', 'Age', 'Glucose', 'BloodPressure', 'DiabetesPedigreeFunction']
X = pima[feature_cols]
y = pima.Outcome
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=1)
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3)