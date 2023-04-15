import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics as sts
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
df
df
df.info()
df.describe()
df['VIP'].value_counts()
print('Linhas: ', df.shape[0])
print('Colunas: ', df.shape[1])
print('\nAtributos : \n', df.columns.tolist())
print('\nValores faltantes :  ', df.isnull().sum().values.sum())
print('\nValores únicos :  \n', df.nunique())
df = df.drop(columns='PassengerId', axis=1)
df[df.isnull().any(axis=1)]
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.NaN, strategy='median')
df['Age'] = imputer.fit_transform(df['Age'].values.reshape(-1, 1))[:, 0]
df['RoomService'] = imputer.fit_transform(df['RoomService'].values.reshape(-1, 1))[:, 0]
df['FoodCourt'] = imputer.fit_transform(df['FoodCourt'].values.reshape(-1, 1))[:, 0]
df['ShoppingMall'] = imputer.fit_transform(df['ShoppingMall'].values.reshape(-1, 1))[:, 0]
df['Spa'] = imputer.fit_transform(df['Spa'].values.reshape(-1, 1))[:, 0]
df['VRDeck'] = imputer.fit_transform(df['VRDeck'].values.reshape(-1, 1))[:, 0]
df.info()
df['CryoSleep'] = df['CryoSleep'].apply(lambda x: 1 if x == 'True' else 0)
df['VIP'] = df['VIP'].apply(lambda x: 1 if x == 'True' else 0)
df = pd.get_dummies(data=df, columns=['HomePlanet'])
df = pd.get_dummies(data=df, columns=['Cabin'])
df = pd.get_dummies(data=df, columns=['Name'])
df = pd.get_dummies(data=df, columns=['Destination'])
df
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Transported'] = le.fit_transform(df['Transported'])
std = StandardScaler()
columns = ['Age', 'RoomService', 'FoodCourt', 'VRDeck', 'ShoppingMall', 'Spa']
scaled = std.fit_transform(df[['Age', 'RoomService', 'FoodCourt', 'VRDeck', 'ShoppingMall', 'Spa']])
scaled = pd.DataFrame(scaled, columns=columns)
df = df.drop(columns=columns, axis=1)
df = df.merge(scaled, left_index=True, right_index=True, how='left')
df
X = df.drop(['Transported'], axis=1).values
y = df['Transported'].values
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=14)
from imblearn.over_sampling import SMOTE
sm = SMOTE()
(x_train_oversampled, y_train_oversampled) = sm.fit_resample(X_train, y_train)
forest = RandomForestClassifier(n_estimators=100)