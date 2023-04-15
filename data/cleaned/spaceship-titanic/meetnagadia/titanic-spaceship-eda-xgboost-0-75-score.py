import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.simplefilter('ignore')
train_data = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_data = pd.read_csv('data/input/spaceship-titanic/test.csv')
sub = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
print(train_data.shape)
print(test_data.shape)
print(sub.shape)
train_data.head()
test_data.head()
sub.head()
print(train_data.dtypes)
print(test_data.dtypes)
train_data.isnull().sum()
test_data.isnull().sum()
train_data['Age'].fillna(train_data['Age'].median(skipna=True), inplace=True)
test_data['Age'].fillna(test_data['Age'].median(skipna=True), inplace=True)
train_data['FoodCourt'].fillna(train_data['FoodCourt'].median(skipna=True), inplace=True)
test_data['FoodCourt'].fillna(test_data['FoodCourt'].median(skipna=True), inplace=True)
train_data['ShoppingMall'].fillna(train_data['ShoppingMall'].median(skipna=True), inplace=True)
test_data['ShoppingMall'].fillna(test_data['ShoppingMall'].median(skipna=True), inplace=True)
train_data['Spa'].fillna(train_data['Spa'].median(skipna=True), inplace=True)
test_data['Spa'].fillna(test_data['Spa'].median(skipna=True), inplace=True)
train_data['VRDeck'].fillna(train_data['VRDeck'].median(skipna=True), inplace=True)
test_data['VRDeck'].fillna(test_data['VRDeck'].median(skipna=True), inplace=True)
train_data['RoomService'].fillna(train_data['RoomService'].median(skipna=True), inplace=True)
test_data['RoomService'].fillna(test_data['RoomService'].median(skipna=True), inplace=True)
train_data['HomePlanet'].value_counts()
train_data['HomePlanet'].fillna('Flat_Earth', inplace=True)
test_data['HomePlanet'].fillna('Flat_Earth', inplace=True)
train_data.isnull().sum()
test_data.isnull().sum()
'CryoSleep:Indicates whether the passenger elected to be put into suspended animation for \n the duration of the voyage. Passengers in cryosleep are confined to their cabins'
train_data.CryoSleep.value_counts()
'Cabin:he cabin number where the passenger is staying. Takes the form deck/num/side, where \nside can be either P for Port or S for Starboard.'
train_data.Cabin.value_counts()
'Destination:The planet the passenger will be debarking to.'
train_data.Destination.value_counts()
'VIP:Whether the passenger has paid for special VIP service during the voyage.'
train_data.VIP.value_counts()
"RoomService: Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities."
train_data.RoomService.value_counts()
'Name: The first and last names of the passenger.'
train_data.Name.value_counts()
train_data = train_data.dropna()
test_data = test_data.dropna()
train_data.isnull().sum()
test_data.isnull().sum()
train_data.columns
test_data.columns
eda_data = train_data
eda_data.shape
eda_data.head()
eda_data.columns
eda_data.drop(['PassengerId', 'Name', 'Cabin', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'], axis=1, inplace=True)
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
hist(train_data['Age'], 'Distribution of Age')
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
train_df = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_df = pd.read_csv('data/input/spaceship-titanic/test.csv')
imputer_cols = ['Age', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'RoomService']
imputer = SimpleImputer(strategy='median')