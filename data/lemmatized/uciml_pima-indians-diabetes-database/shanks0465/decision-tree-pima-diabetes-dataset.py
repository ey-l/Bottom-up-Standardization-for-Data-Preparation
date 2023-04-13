import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.shape
df.head()
df.info()
df.describe()
df.plot(kind='scatter', x='BloodPressure', y='BMI')
print('From df.describe and the plot we can see that few rows have 0 as value for some columns')
zerocols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zerocols:
    df[col] = df[col].replace(0, df[col].mean())
df.plot(kind='scatter', x='BloodPressure', y='BMI')
print('From df.describe and the plot we can see that few rows have 0 as value for some columns')
df.head()
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
corr = df.corr()
pass
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
print('As you can see each of the attributes contribute reasonably towards the outcome')
X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].values
y = df['Outcome']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=0)
clf = DecisionTreeClassifier()