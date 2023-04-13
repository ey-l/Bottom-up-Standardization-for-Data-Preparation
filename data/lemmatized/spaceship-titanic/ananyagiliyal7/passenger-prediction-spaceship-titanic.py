import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head()
_input1.shape
_input1.isna().sum()
_input1.describe().T
categorical_variables = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']
mode = _input1[categorical_variables].mode().iloc[0]
_input1[categorical_variables] = _input1[categorical_variables].fillna(mode)
_input1.isna().sum()
continous_variables = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
median = _input1[continous_variables].median()
median
_input1[continous_variables] = _input1[continous_variables].fillna(median)
_input1.isna().sum()
_input1 = _input1.drop('Name', axis=1)
_input1
_input1[['Deck', 'Num', 'Side']] = _input1['Cabin'].str.split('/', expand=True)
_input1
_input1 = _input1.drop('Cabin', axis=1)
_input1
_input1.hist('Age')
_input1.Age.describe()
labels = ['Child', 'Teenager', 'Adult', 'Older']
bins = [0, 12, 21, 45, 80]
_input1['Age_Group'] = pd.cut(_input1['Age'], bins=bins, labels=labels)
_input1.head()
_input1 = _input1.drop('Age', axis=1)
_input1.head()
lbe = LabelEncoder()
categorical_vars = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Num', 'Side', 'Age_Group', 'Transported']
_input1[categorical_vars] = _input1[categorical_vars].apply(lbe.fit_transform)
_input1
_input1.describe().T
sns.boxplot(data=_input1[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']])
tenth_percentile = np.percentile(_input1['RoomService'], 10)
ninetyseventh_percentile = np.percentile(_input1['RoomService'], 97)
print(f'10% - {tenth_percentile}\n97% - {ninetyseventh_percentile}')
_input1[_input1.RoomService > 1800.0].shape
y = _input1.Transported
X = _input1.drop(['Transported', 'PassengerId'], axis=1)
y.value_counts()
(X_train, X_test, y_train, y_test) = train_test_split(X, y, stratify=y, random_state=127)
print(f'Shape of X_Train: {X_train.shape}\nShape of y_Train: {y_train.shape}\nShape of X_Test: {X_test.shape}\nShape of y_test: {y_test.shape}')
dt_classifier = DecisionTreeClassifier()