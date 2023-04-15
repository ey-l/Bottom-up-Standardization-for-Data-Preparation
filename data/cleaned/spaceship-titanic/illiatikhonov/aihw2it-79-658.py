import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as mn
import contextlib

def cutDec(numObj, digits=0):
    return f'{numObj:.{digits}f}'
train_df = pd.read_csv('data/input/spaceship-titanic/train.csv', index_col='PassengerId')
test_df = pd.read_csv('data/input/spaceship-titanic/test.csv', index_col='PassengerId')
train_df.head(5)
test_df.head(5)
train_df.info()
test_df.info()
mn.matrix(train_df)
mn.matrix(test_df)
df = pd.concat([train_df, test_df], axis=0)
df
mn.matrix(df)
df['HomePlanet'].fillna('Undefined', inplace=True)
df['CryoSleep'].fillna('Undefined', inplace=True)
df['Destination'].fillna('Undefined', inplace=True)
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['VIP'].fillna('Undefined', inplace=True)
df['RoomService'].fillna(df['RoomService'].mean(), inplace=True)
df['FoodCourt'].fillna(df['FoodCourt'].mean(), inplace=True)
df['ShoppingMall'].fillna(df['ShoppingMall'].mean(), inplace=True)
df['Spa'].fillna(df['Spa'].mean(), inplace=True)
df['VRDeck'].fillna(df['VRDeck'].mean(), inplace=True)
df['Name'].fillna('Undefined', inplace=True)
cabindf = df['Cabin'].str.split('/', expand=True)
cabindf.columns = ['Deck', 'Number', 'Side']
df['Deck'] = cabindf['Deck']
df['Number'] = pd.to_numeric(cabindf['Number'], errors='coerce')
df['Side'] = cabindf['Side']
df['Deck'].fillna('Undefined', inplace=True)
df['Number'].fillna(df['Number'].mean(), inplace=True)
df['Side'].fillna('Undefined', inplace=True)
df.drop(['Cabin'], axis=1, inplace=True)
df['PrivateAct'] = df['RoomService'] + df['Spa'] + df['VRDeck']
df['PublicAct'] = df['FoodCourt'] + df['ShoppingMall']
df['Activities'] = df['PrivateAct'] + df['PublicAct']
df['ActCat'] = np.where((df['PrivateAct'] > 0) & (df['PublicAct'] > 0), 'Both', np.where(df['PrivateAct'] > 0, 'Private', np.where(df['PublicAct'] > 0, 'Public', 'No Activity')))
mn.matrix(df)
dfoh = pd.get_dummies(df, columns=['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'ActCat'])
dfoh.drop(['Name'], axis=1, inplace=True)
dfoh
from sklearn.preprocessing import StandardScaler
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Number', 'PrivateAct', 'PublicAct', 'Activities']
dfoh.describe()
dfstd = dfoh.copy()
scaler = StandardScaler()
dfstd[num_cols] = scaler.fit_transform(dfstd[num_cols])
dfstd.describe()
train = dfstd[0:len(train_df)]
test = dfstd[len(train_df):]
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