from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, SVR
import xgboost as xgb
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
del _input1['PassengerId']
del _input1['Name']
_input1
_input1.describe()[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].T.style.background_gradient(cmap='Blues')
_input1.dropna()
print('Number of unique values in each categorical column:')
_input1[['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP', 'Transported']].nunique()
print("Count of unique values in 'Cabin':")
_input1['Cabin'].value_counts()
del _input1['Cabin']
for col in ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Transported']:
    _input1[col] = _input1[col].fillna('Unknown')
fig = make_subplots(rows=3, cols=2, subplot_titles=('HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Transported'), specs=[[{'type': 'domain'}, {'type': 'domain'}], [{'type': 'domain'}, {'type': 'domain'}], [{'type': 'domain'}, {'type': 'domain'}]])
colours = ['#4285f4', '#ea4335', '#fbbc05', '#34a853']
fig.add_trace(go.Pie(labels=np.array(_input1['HomePlanet'].value_counts().index), values=[x for x in _input1['HomePlanet'].value_counts()], textinfo='label+percent', rotation=-45, hole=0.35, marker_colors=colours), row=1, col=1)
fig.add_trace(go.Pie(labels=np.array(_input1['CryoSleep'].value_counts().index), values=[x for x in _input1['CryoSleep'].value_counts()], textinfo='label+percent', hole=0.35, marker_colors=colours), row=1, col=2)
fig.add_trace(go.Pie(labels=np.array(_input1['Destination'].value_counts().index), values=[x for x in _input1['Destination'].value_counts()], textinfo='label+percent', rotation=-45, hole=0.35, marker_colors=colours), row=2, col=1)
fig.add_trace(go.Pie(labels=np.array(_input1['VIP'].value_counts().index), values=[x for x in _input1['VIP'].value_counts()], textinfo='label+percent', rotation=-45, hole=0.35, marker_colors=colours), row=2, col=2)
fig.add_trace(go.Pie(labels=np.array(_input1['Transported'].value_counts().index), values=[x for x in _input1['Transported'].value_counts()], textinfo='label+percent', hole=0.35, marker_colors=colours), row=3, col=1)
fig.update_layout(height=1600, font=dict(size=14), showlegend=False)
fig.show()
fig = px.box(_input1, x='HomePlanet', y='Age', color='Transported')
fig.show()
fig = px.histogram(_input1, x='HomePlanet', color='Transported', color_discrete_map={False: '#ea4335', True: '#4285f4', 'Unknown': '#fbbc05'})
fig.show()
fig = px.histogram(_input1, x='HomePlanet', color='VIP', color_discrete_map={False: '#ea4335', True: '#4285f4', 'Unknown': '#fbbc05'})
fig.show()
fig = px.histogram(_input1, x='Age', color='CryoSleep', marginal='box', color_discrete_map={False: '#ea4335', True: '#4285f4', 'Unknown': '#fbbc05'})
fig.show()
fig = px.histogram(_input1, x='Age', y='ShoppingMall', color='Transported', marginal='box', color_discrete_map={False: '#ea4335', True: '#4285f4', 'Unknown': '#fbbc05'})
fig.show()
fig = px.histogram(_input1, x='Age', y='RoomService', color='Transported', marginal='box', color_discrete_map={False: '#ea4335', True: '#4285f4', 'Unknown': '#fbbc05'})
fig.show()
fig = px.histogram(_input1, x='Age', y='FoodCourt', color='Transported', marginal='box', color_discrete_map={False: '#ea4335', True: '#4285f4', 'Unknown': '#fbbc05'})
fig.show()
fig = px.histogram(_input1, x='Age', y='Spa', color='Transported', marginal='box', color_discrete_map={False: '#ea4335', True: '#4285f4', 'Unknown': '#fbbc05'})
fig.show()
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
del _input1['PassengerId']
del _input1['Name']
del _input1['Cabin']
fig = px.imshow(_input1.isna().transpose(), color_continuous_scale='Blues')
fig.show()
_input1 = _input1.dropna().reset_index(drop=True)
_input1.describe().T[['min', 'max']].style.background_gradient(cmap='Blues')
for col in ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
    _input1[col] = _input1[col] / _input1[col].max()
print('\nCategorical Columns\n')
_input1.select_dtypes(include=['O']).nunique()
for col in ['CryoSleep', 'VIP', 'Transported']:
    le = LabelEncoder()
    _input1[col] = le.fit_transform(_input1[col])
_input1 = pd.get_dummies(_input1, columns=['HomePlanet', 'Destination'], prefix=['HomePlanet', 'Destination'])
features = np.array(_input1[[col for col in _input1.columns if col != 'Transported']])
labels = np.array(_input1['Transported'])
(x_train, x_val, y_train, y_val) = train_test_split(features, labels, test_size=0.2, random_state=0)
model_comparison = {}
parameters = {'C': [6, 8, 10, 12, 14, 16], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
svc_model = SVC()
clf = GridSearchCV(svc_model, parameters)
print('Searching for best hyperparameters ...')