import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.simplefilter('ignore')
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input2 = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
print(_input1.shape)
print(_input0.shape)
print(_input2.shape)
_input1.head()
_input0.head()
_input2.head()
print(_input1.dtypes)
print(_input0.dtypes)
_input1.isnull().sum()
_input0.isnull().sum()
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].median(skipna=True), inplace=False)
_input0['Age'] = _input0['Age'].fillna(_input0['Age'].median(skipna=True), inplace=False)
_input1['FoodCourt'] = _input1['FoodCourt'].fillna(_input1['FoodCourt'].median(skipna=True), inplace=False)
_input0['FoodCourt'] = _input0['FoodCourt'].fillna(_input0['FoodCourt'].median(skipna=True), inplace=False)
_input1['ShoppingMall'] = _input1['ShoppingMall'].fillna(_input1['ShoppingMall'].median(skipna=True), inplace=False)
_input0['ShoppingMall'] = _input0['ShoppingMall'].fillna(_input0['ShoppingMall'].median(skipna=True), inplace=False)
_input1['Spa'] = _input1['Spa'].fillna(_input1['Spa'].median(skipna=True), inplace=False)
_input0['Spa'] = _input0['Spa'].fillna(_input0['Spa'].median(skipna=True), inplace=False)
_input1['VRDeck'] = _input1['VRDeck'].fillna(_input1['VRDeck'].median(skipna=True), inplace=False)
_input0['VRDeck'] = _input0['VRDeck'].fillna(_input0['VRDeck'].median(skipna=True), inplace=False)
_input1['RoomService'] = _input1['RoomService'].fillna(_input1['RoomService'].median(skipna=True), inplace=False)
_input0['RoomService'] = _input0['RoomService'].fillna(_input0['RoomService'].median(skipna=True), inplace=False)
_input1['HomePlanet'].value_counts()
_input1['HomePlanet'] = _input1['HomePlanet'].fillna('Flat_Earth', inplace=False)
_input0['HomePlanet'] = _input0['HomePlanet'].fillna('Flat_Earth', inplace=False)
_input1.isnull().sum()
_input0.isnull().sum()
'CryoSleep:Indicates whether the passenger elected to be put into suspended animation for \n the duration of the voyage. Passengers in cryosleep are confined to their cabins'
_input1.CryoSleep.value_counts()
'Cabin:he cabin number where the passenger is staying. Takes the form deck/num/side, where \nside can be either P for Port or S for Starboard.'
_input1.Cabin.value_counts()
'Destination:The planet the passenger will be debarking to.'
_input1.Destination.value_counts()
'VIP:Whether the passenger has paid for special VIP service during the voyage.'
_input1.VIP.value_counts()
"RoomService: Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities."
_input1.RoomService.value_counts()
'Name: The first and last names of the passenger.'
_input1.Name.value_counts()
_input1 = _input1.dropna()
_input0 = _input0.dropna()
_input1.isnull().sum()
_input0.isnull().sum()
_input1.columns
_input0.columns
eda_data = _input1
eda_data.shape
eda_data.head()
eda_data.columns
eda_data = eda_data.drop(['PassengerId', 'Name', 'Cabin', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'], axis=1, inplace=False)
eda_data.columns
hp = eda_data.HomePlanet.value_counts()
hp
eda_data.HomePlanet.hist(bins=10, color='pink')
plt.title('Home Planet of the Passenagers')
fig = px.pie(eda_data, values=hp.values, names=hp.index, title='Home Planet of the Passengers')
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()
fig = go.Figure([go.Bar(x=hp.index, y=hp.values, text=hp.values, textposition='auto')])
fig.update_layout(title_text='Home Planet of the Passengers')
fig.show()
cs = eda_data.CryoSleep.value_counts()
cs
fig = px.pie(eda_data, values=cs.values, names=cs.index, title='CryoSleep of Passengers')
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()
fig = go.Figure([go.Bar(x=cs.index, y=cs.values, text=cs.values, textposition='auto')])
fig.update_layout(title_text='CryoSleep of the Passengers')
fig.show()
des = eda_data.Destination.value_counts()
des
eda_data.Destination.hist(bins=10, color='pink')
plt.title('Destination of the Passenagers')
fig = px.pie(eda_data, values=des.values, names=des.index, title='Destination of Passengers')
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()
fig = go.Figure([go.Bar(x=des.index, y=des.values, text=des.values, textposition='auto')])
fig.update_layout(title_text='Destination of the Passengers')
fig.show()
age = eda_data.Age.value_counts()
age
eda_data.Age.hist(bins=10, color='yellow')
plt.title('Distribution of Age of the Passenagers')
fig = px.pie(eda_data, values=age.values, names=age.index, title='Age of Passengers')
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()

def hist(col, title):
    plt.figure(figsize=(10, 8))
    ax = sns.histplot(col, kde=False)
    values = np.array([patch.get_height() for patch in ax.patches])
    norm = plt.Normalize(values.min(), values.max())
    colors = plt.cm.rainbow(norm(values))
    for (patch, color) in zip(ax.patches, colors):
        patch.set_color(color)
    plt.title(title, size=20)
hist(_input1['Age'], 'Distribution of Age')
fig = go.Figure([go.Bar(x=age.index, y=age.values, text=age.values, textposition='auto')])
fig.update_layout(title_text='Distribution of Age of the Passengers')
fig.show()
vip = eda_data.VIP.value_counts()
vip
fig = go.Figure([go.Bar(x=vip.index, y=vip.values, text=vip.values, textposition='auto')])
fig.update_layout(title_text='Count of VIP Passengers')
fig.show()
fig = px.pie(eda_data, values=vip.values, names=vip.index, title='Count of VIP Passengers')
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()
tp = eda_data.Transported.value_counts()
tp
fig = go.Figure([go.Bar(x=tp.index, y=tp.values, text=tp.values, textposition='auto')])
fig.update_layout(title_text='Count of Transported Passengers')
fig.show()
fig = px.pie(eda_data, values=tp.values, names=tp.index, title='Count of Transported Passengers')
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
imputer_cols = ['Age', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'RoomService']
imputer = SimpleImputer(strategy='median')