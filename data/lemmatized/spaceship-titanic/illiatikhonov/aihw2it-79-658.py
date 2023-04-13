import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as mn
import contextlib

def cutDec(numObj, digits=0):
    return f'{numObj:.{digits}f}'
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv', index_col='PassengerId')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv', index_col='PassengerId')
_input1.head(5)
_input0.head(5)
_input1.info()
_input0.info()
mn.matrix(_input1)
mn.matrix(_input0)
df = pd.concat([_input1, _input0], axis=0)
df
mn.matrix(df)
df['HomePlanet'] = df['HomePlanet'].fillna('Undefined', inplace=False)
df['CryoSleep'] = df['CryoSleep'].fillna('Undefined', inplace=False)
df['Destination'] = df['Destination'].fillna('Undefined', inplace=False)
df['Age'] = df['Age'].fillna(df['Age'].mean(), inplace=False)
df['VIP'] = df['VIP'].fillna('Undefined', inplace=False)
df['RoomService'] = df['RoomService'].fillna(df['RoomService'].mean(), inplace=False)
df['FoodCourt'] = df['FoodCourt'].fillna(df['FoodCourt'].mean(), inplace=False)
df['ShoppingMall'] = df['ShoppingMall'].fillna(df['ShoppingMall'].mean(), inplace=False)
df['Spa'] = df['Spa'].fillna(df['Spa'].mean(), inplace=False)
df['VRDeck'] = df['VRDeck'].fillna(df['VRDeck'].mean(), inplace=False)
df['Name'] = df['Name'].fillna('Undefined', inplace=False)
cabindf = df['Cabin'].str.split('/', expand=True)
cabindf.columns = ['Deck', 'Number', 'Side']
df['Deck'] = cabindf['Deck']
df['Number'] = pd.to_numeric(cabindf['Number'], errors='coerce')
df['Side'] = cabindf['Side']
df['Deck'] = df['Deck'].fillna('Undefined', inplace=False)
df['Number'] = df['Number'].fillna(df['Number'].mean(), inplace=False)
df['Side'] = df['Side'].fillna('Undefined', inplace=False)
df = df.drop(['Cabin'], axis=1, inplace=False)
df['PrivateAct'] = df['RoomService'] + df['Spa'] + df['VRDeck']
df['PublicAct'] = df['FoodCourt'] + df['ShoppingMall']
df['Activities'] = df['PrivateAct'] + df['PublicAct']
df['ActCat'] = np.where((df['PrivateAct'] > 0) & (df['PublicAct'] > 0), 'Both', np.where(df['PrivateAct'] > 0, 'Private', np.where(df['PublicAct'] > 0, 'Public', 'No Activity')))
mn.matrix(df)
dfoh = pd.get_dummies(df, columns=['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'ActCat'])
dfoh = dfoh.drop(['Name'], axis=1, inplace=False)
dfoh
from sklearn.preprocessing import StandardScaler
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Number', 'PrivateAct', 'PublicAct', 'Activities']
dfoh.describe()
dfstd = dfoh.copy()
scaler = StandardScaler()
dfstd[num_cols] = scaler.fit_transform(dfstd[num_cols])
dfstd.describe()
train = dfstd[0:len(_input1)]
test = dfstd[len(_input1):]
Y = train['Transported'].astype(float)
X = train.drop(['Transported'], axis=1)
Xtest = test.drop(['Transported'], axis=1)
from sklearn.model_selection import train_test_split as tts
(Xtrain, Xval, Ytrain, Yval) = tts(X, Y, test_size=0.1, random_state=10)
Yval = Yval.astype(float)
Ytrain = Ytrain.astype(float)
mn.matrix(Xtrain)
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import accuracy_score
model = lr()