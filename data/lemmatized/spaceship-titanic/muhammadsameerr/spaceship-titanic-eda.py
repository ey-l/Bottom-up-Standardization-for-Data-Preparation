import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.head()
_input0.head()
print(f'\x1b[94m')
print('Shape of training data: ' + str(_input1.shape))
print('Shape of test data: ' + str(_input0.shape))
_input1.iloc[:, :-1].describe().T.sort_values(by='std', ascending=False).style.background_gradient(cmap='GnBu').bar(subset=['max'], color='#BB0000')
object_var = ['Name', 'Cabin', 'Destination']
cat_var = [f for f in _input1.columns if _input1[f].nunique() < 25 and f not in object_var]
cont_var = [f for f in _input1.columns if _input1[f].nunique() > 25 and f not in object_var]
print(f'\x1b[94m')
print('total number of features: ' + str(len(_input1.columns)))
print('Text features: ' + str(len(object_var)))
print('Categorical features: ' + str(len(cat_var)))
print('Continuous features: ' + str(len(cont_var)))
fig = go.Figure(data=[go.Pie(labels=['Categorical', 'Text', 'Continuous'], values=[len(cat_var), len(object_var), len(cont_var)], pull=[0, 0.1, 0], marker=dict(colors=['#FFFF00', '#DE3163', '#58D68D'], line=dict(color='#000000', width=2)))])
fig.show()
print(_input1['Transported'].value_counts())
transported = len(_input1[_input1['Transported'] == True]['Transported'])
not_transported = len(_input1[_input1['Transported'] == False]['Transported'])
print(f'\x1b[94m')
print('Transported: ' + str(transported))
print('Not Transported: ' + str(not_transported))
fig = go.Figure(data=[go.Pie(labels=['Transported', 'Not Transported'], values=[transported, not_transported], pull=[0.1, 0], marker=dict(colors=['#DE3163', '#58D68D'], line=dict(color='#000000', width=2)))])
fig.show()
plt.figure(figsize=(10, 4))
sns.heatmap(_input1.isnull(), yticklabels=False)
plt.title('Null values')
print(f'\x1b[94m')
print(_input1.isnull().sum())
plt.figure(figsize=(10, 4))
sns.histplot(data=_input1, x='Age', hue='Transported', binwidth=1, kde=True)
plt.title('Age distribution')
plt.xlabel('Age')
feature = ['VIP', 'CryoSleep', 'HomePlanet']
fig = plt.figure(figsize=(10, 16))
for (i, f) in enumerate(feature):
    ax = fig.add_subplot(4, 1, i + 1)
    sns.countplot(data=_input1, x=f, axes=ax, hue='Transported')
    ax.set_title(f + ' with Transported')
fig.tight_layout()
cont = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
fig = plt.figure(figsize=(6, 16))
for (i, f) in enumerate(cont):
    ax = fig.add_subplot(5, 1, i + 1)
    sns.histplot(data=_input1, x=f, axes=ax, bins=30, kde=True, hue='Transported')
    plt.ylim([0, 100])
    ax.set_title(f + ' with Transported')
fig.tight_layout()