import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from bokeh.plotting import figure
import plotly.graph_objs as go
import plotly.offline as py
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, auc, accuracy_score
from sklearn.inspection import permutation_importance
import shap
from yellowbrick.model_selection import FeatureImportances
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
diab = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
diab.head()
diab.tail()
healthy = diab[diab['Outcome'] == 0]
diabetic = diab[diab['Outcome'] != 0]
graph = go.Pie(labels=['healthy', 'diabetic'], values=diab['Outcome'].value_counts(), textfont=dict(size=15), opacity=1, marker=dict(colors=['lightskyblue', 'red'], line=dict(color='#000000', width=1)))
fig = dict(data=[graph])
py.iplot(fig)
D_count = diab[['Outcome']].groupby('Outcome').size().reset_index(name='Total')
D_count['% Total'] = D_count['Total'] / sum(D_count['Total']) * 100
D_count
diab.info()
Sante = ['Pregnancies', 'Glucose', 'BloodPressure', 'Insulin', 'Age', 'DiabetesPedigreeFunction', 'BMI', 'Outcome']
sns.heatmap(diab[Sante].corr(), cmap='YlGnBu', annot=True)
plt.title('Feature Correlation Map')
plt.figure(figsize=(15, 10))

from sklearn.model_selection import train_test_split
X = diab.drop(['Outcome'], axis=1)
Y = diab['Outcome']
(x_train, x_test, y_train, y_test) = train_test_split(X, Y, test_size=0.2, random_state=42)
print(x_train.shape)
print(x_test.shape)
diab
from sklearn import ensemble
rf = ensemble.RandomForestClassifier()