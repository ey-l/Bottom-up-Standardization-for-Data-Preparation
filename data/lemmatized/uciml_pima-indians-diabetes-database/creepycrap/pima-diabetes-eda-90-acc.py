import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import pandas_profiling as pp
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df
df.shape
df.info()
df.isnull().sum()
df.describe().T

def hist_count(column, data):
    if column in data:
        pass
        pass
        pass
        pass
for column in df.columns:
    hist_count(column, df)
fig = px.histogram(df, x=df['Pregnancies'], color='Outcome')
pass
fig2 = px.box(df, x=df['Pregnancies'], color='Outcome')
fig2.show()
fig = px.histogram(df, x=df['Glucose'], color='Outcome')
pass
fig2 = px.box(df, x=df['Glucose'], color='Outcome')
fig2.show()
fig = px.histogram(df, x=df['BloodPressure'], color='Outcome')
pass
fig2 = px.box(df, x=df['BloodPressure'], color='Outcome')
fig2.show()
fig = px.histogram(df, x=df['SkinThickness'], color='Outcome')
pass
fig2 = px.box(df, x=df['SkinThickness'], color='Outcome')
fig2.show()
fig = px.histogram(df, x=df['Insulin'], color='Outcome')
pass
fig2 = px.box(df, x=df['Insulin'], color='Outcome')
fig2.show()
fig = px.histogram(df, x=df['BMI'], color='Outcome')
pass
fig2 = px.box(df, x=df['BMI'], color='Outcome')
fig2.show()
fig = px.histogram(df, x=df['DiabetesPedigreeFunction'], color='Outcome')
pass
fig2 = px.box(df, x=df['DiabetesPedigreeFunction'], color='Outcome')
fig2.show()
fig = px.histogram(df, x=df['Age'], color='Outcome')
pass
fig2 = px.box(df, x=df['Age'], color='Outcome')
fig2.show()
fig = px.line(df)
pass
pass
pass
profile = pp.ProfileReport(df, title='Pima Diabetes EDA')
profile
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
x = pd.get_dummies(df[features])
y = df['Outcome']
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=1 / 11, random_state=242)

def Classification_models(x, y, xt, yt):
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import svm
    from sklearn.neighbors import KNeighborsClassifier
    logisreg = LogisticRegression()
    lda = LinearDiscriminantAnalysis()
    gnb = GaussianNB()
    dtc = DecisionTreeClassifier()
    rfc = RandomForestClassifier()
    svmodel = svm.SVC()
    knnmodel = KNeighborsClassifier()