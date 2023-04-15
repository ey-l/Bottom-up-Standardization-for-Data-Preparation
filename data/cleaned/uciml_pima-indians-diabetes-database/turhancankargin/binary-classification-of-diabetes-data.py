import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
diabetes = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
diabetes.head()
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
label = 'Outcome'
(X, y) = (diabetes[features].values, diabetes[label].values)
for n in range(0, 4):
    print('Patient', str(n + 1), '\n  Features:', list(X[n]), '\n  Label:', y[n])
from matplotlib import pyplot as plt

features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
for col in features:
    diabetes.boxplot(column=col, by='Outcome', figsize=(6, 6))
    plt.title(col)

from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=0)
print('Training cases: %d\nTest cases: %d' % (X_train.shape[0], X_test.shape[0]))
from sklearn.linear_model import LogisticRegression
reg = 0.01