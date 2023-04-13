import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
diabetes = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
diabetes.columns
diabetes.info()
diabetes.describe()
diabetes.isnull().sum()
diabetes.head(10)

def check_for_zero(columns):
    for col in columns:
        if 0 in diabetes[col]:
            print(col + ' has 0 in it.')
columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'DiabetesPedigreeFunction', 'Age']
check_for_zero(columns)
X = diabetes.drop('Outcome', axis=1)
Y = diabetes['Outcome']
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=0, strategy='mean')
X[columns] = imp.fit_transform(X[columns])
X.head(10)
X = X.drop('Insulin', axis=1)
import seaborn as sns
import matplotlib.pyplot as plt
pass
X['Pregnancies'].value_counts()

def draw_dist(column):
    pass
    pass
for col in columns:
    draw_dist(col)
pass
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
steps = [('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=5))]
pipeline = Pipeline(steps)
(X_train, X_test, y_train, y_test) = train_test_split(X, Y, test_size=0.4, random_state=42)