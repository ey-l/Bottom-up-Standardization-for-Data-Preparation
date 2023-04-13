import pandas as pd
dataset = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
dataset.head()
feature_cols = ['Pregnancies', 'Insulin', 'BMI', 'Age', 'Glucose', 'BloodPressure', 'DiabetesPedigreeFunction']
x = dataset[feature_cols]
print(x)
y = dataset.Outcome
print(y)
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.25, random_state=7)
print(x_train)
print(y_train)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()