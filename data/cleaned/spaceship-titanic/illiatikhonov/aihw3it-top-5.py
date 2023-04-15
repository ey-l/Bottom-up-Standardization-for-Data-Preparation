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
df['Name'].fillna('UndefName UndefFamName', inplace=True)
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
df['Deck'].fillna('Undefined', inplace=True)
df['Number'].fillna(df['Number'].mean(), inplace=True)
df['Side'].fillna('Undefined', inplace=True)
df['PrivateAct'] = df['RoomService'] + df['Spa'] + df['VRDeck']
df['PublicAct'] = df['FoodCourt'] + df['ShoppingMall']
df['Activities'] = df['PrivateAct'] + df['PublicAct']
df['ActCat'] = np.where((df['PrivateAct'] > 0) & (df['PublicAct'] > 0), 'Both', np.where(df['PrivateAct'] > 0, 'Private', np.where(df['PublicAct'] > 0, 'Public', 'No Activity')))
df.drop(['index', 'FamilyName', 'Cabin', 'Name'], inplace=True, axis=1)
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
train = dfstd[0:len(train_df)]
test = dfstd[len(train_df):]
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