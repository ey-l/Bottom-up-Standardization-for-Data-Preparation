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
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1
_input1
_input1.info()
_input1.describe()
_input1['VIP'].value_counts()
print('Linhas: ', _input1.shape[0])
print('Colunas: ', _input1.shape[1])
print('\nAtributos : \n', _input1.columns.tolist())
print('\nValores faltantes :  ', _input1.isnull().sum().values.sum())
print('\nValores únicos :  \n', _input1.nunique())
_input1 = _input1.drop(columns='PassengerId', axis=1)
_input1[_input1.isnull().any(axis=1)]
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.NaN, strategy='median')
_input1['Age'] = imputer.fit_transform(_input1['Age'].values.reshape(-1, 1))[:, 0]
_input1['RoomService'] = imputer.fit_transform(_input1['RoomService'].values.reshape(-1, 1))[:, 0]
_input1['FoodCourt'] = imputer.fit_transform(_input1['FoodCourt'].values.reshape(-1, 1))[:, 0]
_input1['ShoppingMall'] = imputer.fit_transform(_input1['ShoppingMall'].values.reshape(-1, 1))[:, 0]
_input1['Spa'] = imputer.fit_transform(_input1['Spa'].values.reshape(-1, 1))[:, 0]
_input1['VRDeck'] = imputer.fit_transform(_input1['VRDeck'].values.reshape(-1, 1))[:, 0]
_input1.info()
_input1['CryoSleep'] = _input1['CryoSleep'].apply(lambda x: 1 if x == 'True' else 0)
_input1['VIP'] = _input1['VIP'].apply(lambda x: 1 if x == 'True' else 0)
_input1 = pd.get_dummies(data=_input1, columns=['HomePlanet'])
_input1 = pd.get_dummies(data=_input1, columns=['Cabin'])
_input1 = pd.get_dummies(data=_input1, columns=['Name'])
_input1 = pd.get_dummies(data=_input1, columns=['Destination'])
_input1
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
_input1['Transported'] = le.fit_transform(_input1['Transported'])
std = StandardScaler()
columns = ['Age', 'RoomService', 'FoodCourt', 'VRDeck', 'ShoppingMall', 'Spa']
scaled = std.fit_transform(_input1[['Age', 'RoomService', 'FoodCourt', 'VRDeck', 'ShoppingMall', 'Spa']])
scaled = pd.DataFrame(scaled, columns=columns)
_input1 = _input1.drop(columns=columns, axis=1)
_input1 = _input1.merge(scaled, left_index=True, right_index=True, how='left')
_input1
X = _input1.drop(['Transported'], axis=1).values
y = _input1['Transported'].values
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=14)
from imblearn.over_sampling import SMOTE
sm = SMOTE()
(x_train_oversampled, y_train_oversampled) = sm.fit_resample(X_train, y_train)
forest = RandomForestClassifier(n_estimators=100)