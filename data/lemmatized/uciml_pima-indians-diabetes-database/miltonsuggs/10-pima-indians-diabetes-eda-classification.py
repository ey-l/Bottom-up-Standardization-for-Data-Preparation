import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.express as px
import missingno as msno
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.columns
df.shape
df.reset_index(inplace=True)
df.rename(columns={'index': 'id'}, inplace=True)
df.head()
for col in df.columns:
    df.rename(columns={col: col.lower()}, inplace=True)
df.rename(columns={'bloodpressure': 'blood_pressure', 'skinthickness': 'skin_thickness', 'diabetespedigreefunction': 'diabetes_pedigree_function'}, inplace=True)
df.describe()
df_healthy = df.loc[df['outcome'] == 0]
df_diabetic = df.loc[df['outcome'] == 1]
total = df.isnull().sum().sort_values(ascending=False)
percent = df.isnull().sum() * 100 / df.isnull().count().sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'], sort=False).sort_values('Total', ascending=False)
missing_data.head(40)
df[['glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi']] = df[['glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi']].replace(0, np.NaN)
totals = pd.DataFrame(len(df['id']) - df.isnull().sum(), columns=['count'])
percent = df.isnull().sum() * 100 / df.isnull().count().sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['total', 'percent'], sort=False).sort_values('total', ascending=False)
totals

def missing_plot(dataset, feature):
    totals = pd.DataFrame(len(df['id']) - df.isnull().sum(), columns=['count'])
    missing_percent = df.isnull().sum() * 100 / df.isnull().count().sort_values(ascending=False)
    df_missing = pd.concat([total, missing_percent], axis=1, keys=['total', 'percent'], sort=False).sort_values('total', ascending=False)
    df_missing = df_missing.round(2)
    trace = go.Bar(x=totals.index, y=totals['count'], opacity=0.8, text=df_missing['percent'], textposition='auto', marker=dict(color='#41d9b3', line=dict(color='#000000', width=1.5)))
    layout = dict(title='Missing Value Count & Percentage')
    fig = dict(data=[trace], layout=layout)
    py.iplot(fig)
missing_plot(df, 'id')

def get_mean(feat):
    temp = df[df[feat].notnull()]
    temp = temp[[feat, 'outcome']].groupby(['outcome'])[[feat]].mean().reset_index()
    temp = temp.round(2)
    return temp

def plot_dist(feature, binsize):
    df_healthy = df.loc[df['outcome'] == 0]
    healthy = df[feature]
    df_diabetic = df.loc[df['outcome'] == 1]
    diabetic = df_diabetic[feature]
    hist_data = [healthy, diabetic]
    group_labels = ['healthy', 'diabetic']
    colors = ['#41d9b3', '#c73062']
    fig = ff.create_distplot(hist_data, group_labels, colors=colors, show_hist=True, bin_size=binsize, curve_type='kde')
    fig['layout'].update(title=feature.upper())
    py.iplot(fig, filename='Density plot')
get_mean('insulin')
df.loc[(df['outcome'] == 0) & df['insulin'].isnull(), 'insulin'] = 130.29
df.loc[(df['outcome'] == 1) & df['insulin'].isnull(), 'insulin'] = 206.85
plot_dist('insulin', 0)
get_mean('glucose')
df.loc[(df['outcome'] == 0) & df['glucose'].isnull(), 'glucose'] = 110.64
df.loc[(df['outcome'] == 1) & df['glucose'].isnull(), 'glucose'] = 142.32
plot_dist('glucose', 0)
get_mean('blood_pressure')
df.loc[(df['outcome'] == 0) & df['blood_pressure'].isnull(), 'blood_pressure'] = 70.88
df.loc[(df['outcome'] == 1) & df['blood_pressure'].isnull(), 'blood_pressure'] = 75.32
plot_dist('blood_pressure', 0)
get_mean('skin_thickness')
df.loc[(df['outcome'] == 0) & df['skin_thickness'].isnull(), 'skin_thickness'] = 27.24
df.loc[(df['outcome'] == 1) & df['skin_thickness'].isnull(), 'skin_thickness'] = 33.0
plot_dist('skin_thickness', 0)
get_mean('bmi')
df.loc[(df['outcome'] == 0) & df['bmi'].isnull(), 'bmi'] = 30.86
df.loc[(df['outcome'] == 1) & df['bmi'].isnull(), 'bmi'] = 35.41
plot_dist('bmi', 0)
get_mean('diabetes_pedigree_function')
df.loc[(df['outcome'] == 0) & df['diabetes_pedigree_function'].isnull(), 'diabetes_pedigree_function'] = 0.43
df.loc[(df['outcome'] == 1) & df['diabetes_pedigree_function'].isnull(), 'diabetes_pedigree_function'] = 0.55
plot_dist('diabetes_pedigree_function', 0)
missing_plot(df, 'id')
plot_dist('age', 0)
outcome_preg = df.groupby(['outcome', 'pregnancies'])[['id']].count()
outcome_preg.reset_index(inplace=True)
outcome_preg.rename(columns={'id': 'count'}, inplace=True)
pass
pass
pass
pass
pass
pass
pass

def plot_features(feat1, feat2):
    diabetic = df[df['outcome'] == 1]
    healthy = df[df['outcome'] == 0]
    trace0 = go.Scatter(x=diabetic[feat1], y=diabetic[feat2], name='diabetic', mode='markers', marker=dict(color='#c73062', line=dict(width=1)))
    trace1 = go.Scatter(x=healthy[feat1], y=healthy[feat2], name='healthy', mode='markers', marker=dict(color='#41d9b3', line=dict(width=1)))
    layout = dict(title=feat1.upper() + ' ' + 'vs' + ' ' + feat2.upper(), height=750, width=1000, yaxis=dict(title=feat2.upper(), zeroline=False), xaxis=dict(title=feat1.upper(), zeroline=False))
    plots = [trace0, trace1]
    fig = dict(data=plots, layout=layout)
    py.iplot(fig)

def barplot(feature, sub):
    diabetic = df[df['outcome'] == 1]
    healthy = df[df['outcome'] == 0]
    color = ['#c73062', '#41d9b3']
    trace1 = go.Bar(x=diabetic[feature].value_counts().keys().tolist(), y=diabetic[feature].value_counts().values.tolist(), text=diabetic[feature].value_counts().values.tolist(), textposition='auto', name='diabetic', opacity=0.8, marker=dict(color='#c73062', line=dict(color='#000000', width=1)))
    trace2 = go.Bar(x=healthy[feature].value_counts().keys().tolist(), y=healthy[feature].value_counts().values.tolist(), text=healthy[feature].value_counts().values.tolist(), textposition='auto', name='healthy', opacity=0.8, marker=dict(color='#41d9b3', line=dict(color='#000000', width=1)))
    layout = dict(title=str(feature) + ' ' + sub, xaxis=dict(), yaxis=dict(title='Count'), yaxis2=dict(range=[-0, 75], overlaying='y', anchor='x', side='right', zeroline=False, showgrid=False, title='% diabetic'))
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    py.iplot(fig)

def pieplot(feature, sub):
    diabetic = df[df['outcome'] == 1]
    healthy = df[df['outcome'] == 0]
    col = ['Silver', 'mediumturquoise', '#CF5C36', 'lightblue', 'magenta', '#FF5D73', '#F2D7EE', 'mediumturquoise']
    trace1 = go.Pie(values=diabetic[feature].value_counts().values.tolist(), labels=diabetic[feature].value_counts().keys().tolist(), textfont=dict(size=15), opacity=0.8, hole=0.5, hoverinfo='label+percent+name', domain=dict(x=[0.0, 0.48]), name='Diabetic', marker=dict(colors=col, line=dict(width=1.5)))
    trace2 = go.Pie(values=healthy[feature].value_counts().values.tolist(), labels=healthy[feature].value_counts().keys().tolist(), textfont=dict(size=15), opacity=0.8, hole=0.5, hoverinfo='label+percent+name', marker=dict(line=dict(width=1.5)), domain=dict(x=[0.52, 1]), name='Healthy')
    layout = go.Layout(dict(title=feature.upper() + ' distribution by target: ' + sub, annotations=[dict(text='Diabetic' + ' : ' + '268', font=dict(size=13), showarrow=False, x=0.22, y=-0.1), dict(text='Healthy' + ' : ' + '500', font=dict(size=13), showarrow=False, x=0.8, y=-0.1)]))
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    py.iplot(fig)
outcome = df.groupby(['outcome'])[['id']].count()
outcome.reset_index(inplace=True)
outcome.rename(columns={'id': 'count'}, inplace=True)
outcome.sort_values(by='count', ascending=False, inplace=True)
outcome
pass
pass
pass
pass
pass
pass
pass
pass
plot_features('pregnancies', 'age')
df.loc[:, 'feat1'] = 0
df.loc[(df['age'] <= 30) & (df['pregnancies'] <= 6), 'feat1'] = 1
barplot('feat1', ':AGE <= 30 & PREGNANCIES <= 6')
pieplot('feat1', 'AGE <= 30 & PREGNANCIES <= 6')
plot_features('bmi', 'age')
df.loc[:, 'feat2'] = 0
df.loc[(df['age'] <= 30) & (df['bmi'] <= 30), 'feat2'] = 1
barplot('feat2', ': AGE <= 30 & BMI <= 30')
pieplot('feat2', 'AGE <= 30 & BMI <= 30')
plot_features('skin_thickness', 'age')
df.loc[:, 'feat3'] = 0
df.loc[(df['age'] <= 30) & (df['skin_thickness'] <= 32), 'feat3'] = 1
barplot('feat3', ': AGE <=30 & SKIN THICKNESS <=32')
pieplot('feat3', 'AGE <=30 & SKIN THICKNESS <=32')
plot_features('glucose', 'age')
df.loc[:, 'feat4'] = 0
df.loc[(df['age'] <= 30) & (df['glucose'] <= 120), 'feat4'] = 1
barplot('feat4', ': AGE <=30 & GLUCOSE <=120')
pieplot('feat4', 'AGE <= 30 & GLUCOSE <= 120')
plot_features('glucose', 'blood_pressure')
df.loc[:, 'feat5'] = 0
df.loc[(df['glucose'] <= 100) & (df['blood_pressure'] <= 80), 'feat5'] = 1
barplot('feat5', ': GLUCOSE <= 100 & BLOOD PRESSURE <=80')
pieplot('feat5', 'GLUCOSE <= 100 & BLOOD PRESSURE <=80')
plot_features('glucose', 'bmi')
df.loc[:, 'feat6'] = 0
df.loc[(df['bmi'] <= 40) & (df['glucose'] <= 100), 'feat6'] = 1
barplot('feat6', ': GLUCOSE <= 100 & BMI <= 40')
pieplot('feat6', 'GLUCOSE <= 100 & BMI <= 40')
plot_features('glucose', 'skin_thickness')
df.loc[:, 'feat7'] = 0
df.loc[(df['glucose'] <= 120) & (df['skin_thickness'] <= 32), 'feat7'] = 1
barplot('feat7', ': GLUCOSE <= 120 & SKIN THICKNESS <= 32')
pieplot('feat7', 'GLUCOSE <= 120 & SKIN THICKNESS <= 32')
plot_features('glucose', 'insulin')
df.loc[:, 'feat8'] = 0
df.loc[(df['insulin'] <= 130) & (df['glucose'] <= 120), 'feat8'] = 1
barplot('feat8', ': GLUCOSE <= 120 & INSULIN <= 130')
pieplot('feat8', 'GLUCOSE <= 120 & INSULIN <= 130')
plot_features('blood_pressure', 'bmi')
df.loc[:, 'feat9'] = 0
df.loc[(df['bmi'] <= 30) & (df['blood_pressure'] <= 80), 'feat9'] = 1
barplot('feat9', ': BMI <= 30 & BLOOD PRESSURE <= 80')
barplot('feat9', 'BMI <= 30 & BLOOD PRESSURE <= 80')
plot_features('blood_pressure', 'skin_thickness')
df.loc[:, 'feat10'] = 0
df.loc[(df['blood_pressure'] <= 80) & (df['skin_thickness'] <= 28), 'feat10'] = 1
barplot('feat10', ': BLOOD PRESSURE <= 80 & SKIN THICKNESS <= 28')
pieplot('feat10', 'BLOOD PRESSURE <= 80 & SKIN THICKNESS <= 28')
plot_features('skin_thickness', 'insulin')
df.loc[:, 'feat11'] = 0
df.loc[(df['skin_thickness'] <= 40) & (df['insulin'] <= 131), 'feat11'] = 1
barplot('feat11', ': SKIN THICKNESS <= 28 & INSULIN <= 131')
pieplot('feat11', 'SKIN THICKNESS <= 28 & INSULIN <= 131')
plot_features('skin_thickness', 'bmi')
df.loc[:, 'feat12'] = 0
df.loc[(df['bmi'] <= 30) & (df['skin_thickness'] <= 28), 'feat12'] = 1
barplot('feat12', ': SKIN THICKNESS <= 28 & BMI <= 30')
pieplot('feat12', 'SKIN THICKNESS <= 28 & BMI <= 30')
plot_features('insulin', 'bmi')
df.loc[:, 'feat13'] = 0
df.loc[(df['bmi'] <= 40) & (df['insulin'] <= 131), 'feat13'] = 1
barplot('feat13', ': BMI <= 40 & INSULIN <= 131')
pieplot('feat13', 'BMI <= 40 & INSULIN <= 131')
pass
pass
pass