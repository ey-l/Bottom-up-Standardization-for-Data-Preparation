import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data
plt.figure(figsize=(12, 10))
p = sns.heatmap(data.corr(), annot=True, cmap='RdYlGn')
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = pd.DataFrame(sc_X.fit_transform(data.drop(['Outcome'], axis=1)), columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
X.head()
y = data.Outcome
y
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.33, random_state=7)
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()