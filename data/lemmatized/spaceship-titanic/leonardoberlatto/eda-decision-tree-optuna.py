import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px
import plotly.graph_objects as go
plotly.offline.init_notebook_mode(connected=True)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0['Transported'] = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')['Transported']
dataset = _input1.append(_input0).set_index('PassengerId')
dataset.head()
dataset.info()
dataset.describe().T
plt.figure(figsize=(10, 8))
passengers_by_survivence = dataset['Transported'].value_counts()
passengers_by_survivence = passengers_by_survivence.rename_axis('Transported').reset_index(name='Counts')
labels = ['No', 'Yes']
fig = px.pie(passengers_by_survivence, values='Counts', names=labels, title='Passergers Transported Percentage', color_discrete_sequence=px.colors.qualitative.G10)
fig.show(renderer='kaggle')
plt.figure(figsize=(10, 8))
fig = px.histogram(dataset, x='Age', title='Age Distribuition', histnorm='', color_discrete_sequence=px.colors.qualitative.G10)
fig.show(renderer='kaggle')
fig = px.histogram(dataset, x='Age', color='Transported', title='Passengers ages by Transported condition', histnorm='', color_discrete_sequence=px.colors.qualitative.G10)
fig.show(renderer='kaggle')
dataset['VIP'] = dataset['VIP'].fillna(False)
dataset['CryoSleep'] = dataset['CryoSleep'].fillna(False)
plt.figure(figsize=(10, 8))
survivor_count_per_sex = px.histogram(dataset, x='VIP', color='Transported', title='Passengers Transported by VIP or not', barmode='group', color_discrete_sequence=px.colors.qualitative.G10)
survivor_count_per_sex.show(renderer='kaggle')
dataset['CryoSleep'] = dataset['CryoSleep'].fillna(False)
plt.figure(figsize=(10, 8))
survivor_count_per_sex = px.histogram(dataset, x='CryoSleep', color='Transported', title='Passengers Transported by Cryosleep Condition', barmode='group', color_discrete_sequence=px.colors.qualitative.G10)
survivor_count_per_sex.show(renderer='kaggle')
dataset['HomePlanet'].value_counts()
plt.figure(figsize=(10, 8))
passengers_by_survivence = dataset['HomePlanet'].value_counts()
passengers_by_survivence = passengers_by_survivence.rename_axis('HomePlanet').reset_index(name='Counts')
values = dataset['HomePlanet'].value_counts().keys().tolist()
labels = ['No', 'Yes']
fig = px.pie(passengers_by_survivence, values='Counts', names=values, title='Passergers Transported by Home Planet', color_discrete_sequence=px.colors.qualitative.G10)
fig.show(renderer='kaggle')
dataset['Cabin'] = dataset['Cabin'].fillna('')
dataset['Cabin'].str.split('/', expand=True).info()
from sklearn.impute import SimpleImputer
dataset[['deck', 'number', 'side']] = dataset['Cabin'].str.split('/', expand=True)
most_frequent_imputer = SimpleImputer(missing_values=None, strategy='most_frequent')