import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1
_input2 = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
_input2
_input1.head(10)
_input1.info()
del _input1['PassengerId']
del _input1['Cabin']
del _input1['Destination']
del _input1['ShoppingMall']
del _input1['Name']
del _input1['HomePlanet']
_input1.describe()
from sklearn.preprocessing import LabelEncoder
L = LabelEncoder()
_input1['CryoSleep'] = L.fit_transform(_input1['CryoSleep'])
_input1['VIP'] = L.fit_transform(_input1['VIP'])
_input1['Transported'] = L.fit_transform(_input1['Transported'])
_input1
_input1.isnull().sum()
_input1['Age'].mean()
_input1['RoomService'].mean()
_input1['FoodCourt'].mean()
_input1['Spa'].mean()
_input1['VRDeck'].mean()
_input1['Age'] = _input1['Age'].fillna(28)
_input1['RoomService'] = _input1['RoomService'].fillna(224)
_input1['FoodCourt'] = _input1['FoodCourt'].fillna(458)
_input1['Spa'] = _input1['Spa'].fillna(311)
_input1['VRDeck'] = _input1['VRDeck'].fillna(304)
_input1
Q1 = _input1.quantile(0.25)
Q3 = _input1.quantile(0.75)
IQR = Q3 - Q1
IQR
print('outlier Counter of the all features')
((_input1 < Q1 - 1.5 * IQR) | (_input1 > Q3 + 1.5 * IQR)).sum()
for col in _input1.columns:
    if _input1[col].dtypes != 'object':
        (q1, q3) = (_input1[col].quantile(0.25), _input1[col].quantile(0.75))
        iqr = q3 - q1
        ll = q1 - 1.5 * iqr
        ul = q3 + 1.5 * iqr
        _input1['Age'] = np.where(_input1['Age'] > ul, _input1['Age'].mean(), np.where(_input1['Age'] < ll, _input1['Age'].mean(), _input1['Age']))
        _input1['VIP'] = np.where(_input1['VIP'] > ul, _input1['VIP'].mean(), np.where(_input1['VIP'] < ll, _input1['VIP'].mean(), _input1['VIP']))
        _input1['RoomService'] = np.where(_input1['RoomService'] > ul, _input1['RoomService'].mean(), np.where(_input1['RoomService'] < ll, _input1['RoomService'].mean(), _input1['RoomService']))
        _input1['FoodCourt'] = np.where(_input1['FoodCourt'] > ul, _input1['FoodCourt'].mean(), np.where(_input1['FoodCourt'] < ll, _input1['FoodCourt'].mean(), _input1['Age']))
        _input1['Spa'] = np.where(_input1['Spa'] > ul, _input1['Spa'].mean(), np.where(_input1['Spa'] < ll, _input1['Spa'].mean(), _input1['Spa']))
        _input1['VRDeck'] = np.where(_input1['VRDeck'] > ul, _input1['VRDeck'].mean(), np.where(_input1['VRDeck'] < ll, _input1['VRDeck'].mean(), _input1['VRDeck']))
Q1 = _input1.quantile(0.25)
Q3 = _input1.quantile(0.75)
IQR = Q3 - Q1
IQR
print('outlier Counter of the all features')
((_input1 < Q1 - 1.5 * IQR) | (_input1 > Q3 + 1.5 * IQR)).sum()
_input1.isnull().sum()
_input1['Transported'].value_counts()
x = _input1.iloc[:, :-1].values
y = _input1.iloc[:, -1].values
y
from sklearn.preprocessing import StandardScaler
S = StandardScaler()
X = S.fit_transform(x)
X
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=25)
from collections import Counter
Counter(y_train)
from sklearn.ensemble import RandomForestClassifier
R = RandomForestClassifier()