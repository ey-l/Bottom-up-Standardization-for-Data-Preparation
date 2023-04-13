import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df
df.info()
df.describe()
import plotly
import plotly.express as px
px.histogram(df, x='Pregnancies', color='Outcome', width=350, height=250)
px.histogram(df, x='Glucose', color='Outcome', width=350, height=250)
px.histogram(df, x='BloodPressure', color='Outcome', width=350, height=250)
px.histogram(df, x='SkinThickness', color='Outcome', width=350, height=250)
px.histogram(df, x='Insulin', color='Outcome', width=350, height=250)
px.histogram(df, x='BMI', color='Outcome', width=350, height=250)
px.histogram(df, x='DiabetesPedigreeFunction', color='Outcome', width=350, height=250)
px.histogram(df, x='Age', color='Outcome', width=350, height=250)
X = df.drop(['Outcome'], axis=1)
y = df['Outcome']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, np.array(y), test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
LR_model = LogisticRegression(random_state=42)
LR_model_params = {'solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'], 'penalty': ['l1', 'l2', 'elasticnet', None], 'C': list(np.random.random_sample((5,))), 'class_weight': ['balanced', None], 'warm_start': [True, False]}
LR_GS = GridSearchCV(LR_model, LR_model_params, scoring=['accuracy', 'f1'], refit='f1', cv=5, verbose=4)