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
df['Name'] = df['Name'].fillna('UndefName UndefFamName', inplace=False)
df['index'] = df.index
df[['ID']] = df['index'].str.split('_', expand=True)[[0]].astype(int)
df['GroupSize'] = df.groupby(['ID'])['ID'].transform('count')
df['Child'] = df['Age'].apply(lambda x: 1 if x < 14 else 0)
df['Youth'] = df['Age'].apply(lambda x: 1 if x in range(14, 27) else 0)
df['Middle'] = df['Age'].apply(lambda x: 1 if x in range(28, 41) else 0)
df['Old'] = df['Age'].apply(lambda x: 1 if x in range(42, 63) else 0)
df['VeryOld'] = df['Age'].apply(lambda x: 1 if x > 64 else 0)
df['FamilyName'] = df['Name'].str.split(' ', expand=True)[1]
familySize = df.pivot_table(columns=['FamilyName'], aggfunc='size')
familySize = familySize.reset_index()
familySize.columns = ['FamilyName', 'Size']

def findID(datafile):
    return familySize['Size'][np.where(familySize['FamilyName'] == datafile['FamilyName'])[0][0]]
df['FamilyAmount'] = df.apply(findID, axis=1)
del familySize
cabindf = df['Cabin'].str.split('/', expand=True)
cabindf.columns = ['Deck', 'Number', 'Side']
df['Deck'] = cabindf['Deck']
df['Number'] = pd.to_numeric(cabindf['Number'], errors='coerce')
df['Side'] = cabindf['Side']
df['Deck'] = df['Deck'].fillna('Undefined', inplace=False)
df['Number'] = df['Number'].fillna(df['Number'].mean(), inplace=False)
df['Side'] = df['Side'].fillna('Undefined', inplace=False)
df['PrivateAct'] = df['RoomService'] + df['Spa'] + df['VRDeck']
df['PublicAct'] = df['FoodCourt'] + df['ShoppingMall']
df['Activities'] = df['PrivateAct'] + df['PublicAct']
df['ActCat'] = np.where((df['PrivateAct'] > 0) & (df['PublicAct'] > 0), 'Both', np.where(df['PrivateAct'] > 0, 'Private', np.where(df['PublicAct'] > 0, 'Public', 'No Activity')))
df = df.drop(['index', 'FamilyName', 'Cabin', 'Name'], inplace=False, axis=1)
mn.matrix(df)
dfoh = pd.get_dummies(df, columns=['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'ActCat', 'Child', 'Youth', 'Middle', 'Old', 'VeryOld'])
dfoh
from sklearn.preprocessing import StandardScaler
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Number', 'PrivateAct', 'PublicAct', 'Activities', 'GroupSize', 'ID', 'FamilyAmount']
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
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
kfold = KFold(n_splits=2, shuffle=True, random_state=21)
valScores = []
trainScores = []
estimators = []
for (trIdx, valIdx) in kfold.split(X):
    (Xtr, Xval) = (X.iloc[trIdx], X.iloc[valIdx])
    (Ytr, Yval) = (Y.iloc[trIdx], Y.iloc[valIdx])
    model = GridSearchCV(CatBoostClassifier(), param_grid={'iterations': range(200, 2000, 100), 'eval_metric': ['Accuracy'], 'verbose': [0]}, n_jobs=-1, cv=3)