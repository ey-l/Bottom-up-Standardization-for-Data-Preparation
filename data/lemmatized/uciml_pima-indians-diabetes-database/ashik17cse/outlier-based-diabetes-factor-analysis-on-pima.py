import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
db = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
print(db.columns)
db
db.shape
db.info()
db.isnull().values.any()
import seaborn as sns
print(db.groupby('Outcome').size())
pass
columns = db.columns[:8]
pass
length = len(columns)
for (i, j) in itertools.zip_longest(columns, range(length)):
    pass
    pass
    db[i].hist(bins=20, edgecolor='black')
    pass
pass
db.corr()
corr_val = db.corr()
top_corr_features = corr_val.index
pass
pass
import plotly.express as px
df = db
fig = px.scatter_3d(df, x='Glucose', y='DiabetesPedigreeFunction', z='Outcome', color='Outcome')
pass
import plotly.express as px
df = db
fig = px.scatter_3d(df, x='Glucose', y='BloodPressure', z='Outcome', color='Outcome')
pass
import plotly.express as px
df = db
fig = px.scatter_3d(df, x='Outcome', y='Insulin', z='Glucose', color='Outcome')
pass
for template in ['plotly']:
    fig = px.scatter(db, x='Glucose', y='BloodPressure', color='Outcome', log_x=True, size_max=20, template=template, title='Diabetes or not!')
pass
for template in ['plotly']:
    fig = px.scatter(db, x='Glucose', y='DiabetesPedigreeFunction', color='Outcome', log_x=True, size_max=20, template=template, title='Diabetes or not!')
pass
for template in ['plotly']:
    fig = px.scatter(db, x='Glucose', y='BMI', color='Outcome', log_x=True, size_max=20, template=template, title='Diabetes or not!')
pass
for template in ['plotly']:
    fig = px.scatter(db, x='Glucose', y='Insulin', color='Outcome', log_x=True, size_max=20, template=template, title='Diabetes or not!')
pass
pass
pass
pass
pass
pass
pass
pass
pass
X_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
Y_cols = ['Outcome']
x = db[X_cols]
y = db[Y_cols]
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(x, y, test_size=0.2, random_state=66)
X_train.shape
X_train
from sklearn.metrics import accuracy_score

def model_Evaluate(model):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    cf_matrix = confusion_matrix(y_test, y_pred)
    categories = ['Negative', 'Positive']
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f'{v1}\n{v2}' for (v1, v2) in zip(group_names, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    pass
    pass
    pass
from sklearn.linear_model import LogisticRegression