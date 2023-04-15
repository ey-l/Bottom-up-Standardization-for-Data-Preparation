import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
df_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
df_train.head(4).style.background_gradient()
print(f'Train Data Shape: {df_train.shape}')
print(f'Test Data Shape: {df_test.shape}')
df_train.describe().style.background_gradient()
Transported_data = df_train.Transported
df_combined = pd.concat([df_train.drop('Transported', axis=1), df_test], axis=0)
print(f'Shape of the combined dataset: {df_combined.shape}')
df_combined.isnull().sum()
df_combined1 = df_combined.copy()
categorical = df_combined1.columns[df_combined1.dtypes == 'object']
print(f'Columns with categorical data:\n {categorical}\n')
numerical = df_combined1.columns[df_combined1.dtypes != 'object']
print(f'Columns with numerical data:\n {numerical}')
x = df_combined1.HomePlanet.mode()
y = x.to_string()
print(f'String before: {y}')
y_converted_to_string = y.strip('0 ')
print(f'String after: {y_converted_to_string}')
for i in categorical:
    if i != 'CryoSleep':
        x = df_combined1[i].mode()
        y = x.to_string()
        y_converted_to_string = y.strip('0 ')
        df_combined1[i].fillna(y_converted_to_string, inplace=True)
for i in numerical:
    df_combined1[i].fillna(df_combined1[i].median(), inplace=True)
df_combined1.isnull().sum()
df_combined1.CryoSleep.mode()
df_combined1.CryoSleep.fillna(bool(0), inplace=True)
df_combined1.isnull().sum()
import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as cf
import plotly.express as px
cf.go_offline()
df_visualize = df_combined1[:8693]
df_visualize['Transported'] = Transported_data
fig = px.histogram(df_visualize, y='HomePlanet', color='Transported', color_discrete_map={True: 'MediumPurple', False: 'lightblue'}, width=700, height=400)
fig.update_layout(template='plotly_dark', title='Original Planet and Transported People', font=dict(family='PT Sans', size=14))
fig
fig = px.histogram(df_visualize, y='CryoSleep', color='Transported', color_discrete_map={True: 'seagreen', False: 'yellowgreen'}, width=700, height=400)
fig.update_layout(template='plotly_dark', title='Cryosleep and Transported People Analysis', font=dict(family='PT Sans', size=14), yaxis_title='Was the passenger in Cryosleep?', xaxis_title='Number of People')
fig
fig = px.histogram(df_visualize, y='Destination', color='Transported', color_discrete_map={True: 'saddlebrown', False: 'lightcoral'}, width=700, height=400)
fig.update_layout(template='plotly_dark', title='Destination and Transported people analysis', font=dict(family='PT Sans', size=14), yaxis_title='Destination', xaxis_title='Number of People')
fig
fig = px.box(df_visualize, y='Age', color='Transported', color_discrete_map={True: 'MediumPurple', False: 'lightblue'}, points='all', title='Distribution of the Age', width=700, height=400)
fig.update_layout(template='plotly_dark', font=dict(family='Sans', size=17))
fig.update_traces(marker=dict(size=0.75))
fig
df_visualize['Recreational_activities'] = df_visualize['RoomService'] + df_visualize['Spa']
+df_visualize['FoodCourt'] + df_visualize['ShoppingMall'] + df_visualize['VRDeck']
fig = px.box(df_visualize, x='Recreational_activities', color='Transported', color_discrete_map={True: 'orchid', False: 'lightblue'}, points='all', title='Distribution of the expenditure on Room Service', width=700, height=400)
fig.update_layout(template='plotly_dark', font=dict(family='Sans', size=17), xaxis_title='Expenditure on Recreational Activities')
fig.update_traces(marker=dict(size=0.75))
fig
df_combined1.head(2).style.background_gradient()
Passenger_Id_of_Test_data = df_combined1.PassengerId[8693:]
df_combined1.PassengerId = df_combined1.PassengerId.apply(lambda x: x[:4])
df_combined1.PassengerId[:10]
df_combined1.drop(['Name', 'Cabin', 'PassengerId'], axis=1, inplace=True)
df_combined1.head(2)
df_final = pd.get_dummies(df_combined1, columns=categorical.drop(labels=['Name', 'Cabin', 'PassengerId']), drop_first=True)
df_final.head()
df_final.shape
train_data = df_final[:8693]
X = train_data
y = Transported_data
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(train_data, y, test_size=0.25, random_state=12)
from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression(max_iter=10000)