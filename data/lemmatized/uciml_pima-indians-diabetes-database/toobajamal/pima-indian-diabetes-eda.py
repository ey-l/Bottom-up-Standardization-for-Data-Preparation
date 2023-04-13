import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.info()
pass
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'
df[['Glucose', 'BMI', 'BloodPressure', 'Insulin', 'SkinThickness']] = df[['Glucose', 'BMI', 'BloodPressure', 'Insulin', 'SkinThickness']].replace(0, np.nan)
df.isnull().sum()
for col in ['Glucose', 'BMI', 'BloodPressure', 'Insulin', 'SkinThickness']:
    df[col] = df[col].fillna(df[col].mean())
df.describe()
is_diabetic = {0: 'No', 1: 'Yes'}
df['is_diabetic'] = df.Outcome.map(is_diabetic)
fig = px.histogram(df, x='Pregnancies', marginal='box', color='is_diabetic', nbins=18, title='Distribution of Pregnancies')
fig.update_layout(bargap=0.1)
pass
df.Glucose.describe()
fig = px.histogram(df, x='Glucose', marginal='box', color='is_diabetic', nbins=100, title='Distribution of Glucose')
fig.update_layout(bargap=0.1)
pass
fig = px.histogram(df, x='BloodPressure', marginal='box', color='is_diabetic', title='Distribution of BloodPressure')
fig.update_layout(bargap=0.1)
pass
fig = px.histogram(df, x='SkinThickness', marginal='box', color='is_diabetic', nbins=50, title='Distribution of Skin Thickness')
fig.update_layout(bargap=0.1)
pass
df.Insulin.describe()
fig = px.histogram(df, x='Insulin', marginal='box', nbins=20, color='is_diabetic', title='Insulin')
fig.update_layout(bargap=0.1)
pass
fig = px.histogram(df, x='BMI', marginal='box', color='is_diabetic', nbins=50, title='Distribution of BMI')
fig.update_layout(bargap=0.1)
pass
fig = px.histogram(df, x='DiabetesPedigreeFunction', marginal='box', color='is_diabetic', nbins=5, title='Distribution of Diabetes Pedigree Function')
fig.update_layout(bargap=0.1)
pass
fig = px.histogram(df, x='Age', marginal='box', color='is_diabetic', title='Distribution of Outcome with respect to Age')
fig.update_layout(bargap=0.1)
pass
corr = df.corr()
pass