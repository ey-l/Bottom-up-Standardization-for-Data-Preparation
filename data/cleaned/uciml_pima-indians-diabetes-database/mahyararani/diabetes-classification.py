import numpy as pd
import pandas as pd
import matplotlib.pyplot as plt

diabetes = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
diabetes.head()
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
label = 'Outcome'
(X, y) = (diabetes[features].values, diabetes[label].values)
for col in features:
    diabetes.boxplot(column=col, by='Outcome', figsize=(6, 6))
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=0)
print('Training cases: %d\n Test cases: %d' % (X_train.shape[0], X_test.shape[0]))
from sklearn.linear_model import LogisticRegression
reg = 0.01