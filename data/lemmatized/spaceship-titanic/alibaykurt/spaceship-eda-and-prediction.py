from unicodedata import category
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.info()
_input1.describe()
_input1 = _input1.drop('Name', axis=1)
_input0 = _input0.drop('Name', axis=1)
p_id = _input0['PassengerId']

def bar_plot(variable):
    feature = _input1[variable]
    featureValue = feature.value_counts()
    plt.bar(featureValue.index, featureValue)
    plt.xticks(featureValue.index, featureValue.index.values)
    plt.ylabel('Frequency')
    plt.title(variable)
    plt.figure(figsize=(6, 3))
category1 = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Transported']
for i in category1:
    bar_plot(i)
_input1['HomePlanet'] = _input1['HomePlanet'].fillna('Earth')
_input0['HomePlanet'] = _input0['HomePlanet'].fillna('Earth')
_input1['CryoSleep'] = _input1['CryoSleep'].fillna(False)
_input0['CryoSleep'] = _input0['CryoSleep'].fillna(False)
cabintemp1 = _input1['Cabin'].str.split(pat='/', expand=True)
_input1[['deck', 'temp', 'side']] = cabintemp1
cabintemp2 = _input0['Cabin'].str.split(pat='/', expand=True)
_input0[['deck', 'temp', 'side']] = cabintemp2
_input1['deck'] = _input1['deck'].fillna('F')
_input1['side'] = _input1['side'].fillna('S')
_input0['deck'] = _input0['deck'].fillna('F')
_input0['side'] = _input0['side'].fillna('S')
_input1['Destination'] = _input1['Destination'].fillna('TRAPPIST-1e')
_input0['Destination'] = _input0['Destination'].fillna('TRAPPIST-1e')
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].mean())
_input0['Age'] = _input0['Age'].fillna(_input1['Age'].mean())
_input1['VIP'] = _input1['VIP'].fillna(False)
_input0['VIP'] = _input0['VIP'].fillna(False)
_input1[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = _input1[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(0.0)
_input0[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = _input0[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(0.0)
_input1 = _input1.drop(['Cabin', 'temp', 'PassengerId'], axis=1)
_input0 = _input0.drop(['Cabin', 'temp', 'PassengerId'], axis=1)
_input1 = pd.get_dummies(_input1, columns=['HomePlanet'])
_input0 = pd.get_dummies(_input0, columns=['HomePlanet'])
_input1 = pd.get_dummies(_input1, columns=['Destination'])
_input0 = pd.get_dummies(_input0, columns=['Destination'])
_input1 = pd.get_dummies(_input1, columns=['deck'])
_input0 = pd.get_dummies(_input0, columns=['deck'])
_input1 = pd.get_dummies(_input1, columns=['side'])
_input0 = pd.get_dummies(_input0, columns=['side'])
_input1 = _input1.replace({False: 0, True: 1}, inplace=False)
_input0 = _input0.replace({False: 0, True: 1}, inplace=False)
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
y = _input1['Transported']
x = _input1.drop(['Transported'], axis=1)
(X_train, X_test, y_train, y_test) = train_test_split(x, y, test_size=0.33, random_state=42)
random_state = 42
classifier = [DecisionTreeClassifier(random_state=random_state), SVC(random_state=random_state), RandomForestClassifier(random_state=random_state), LogisticRegression(random_state=random_state), KNeighborsClassifier()]
dt_param_grid = {'min_samples_split': range(10, 500, 20), 'max_depth': range(1, 20, 2)}
svc_param_grid = {'kernel': ['rbf'], 'gamma': [0.001, 0.01, 0.1, 1], 'C': [1, 10, 100, 1000]}
rf_param_grid = {'max_features': [1, 3, 10], 'min_samples_split': [2, 3, 10], 'min_samples_leaf': [1, 3, 10], 'bootstrap': [False], 'n_estimators': [100, 300], 'criterion': ['gini']}
logreg_param_grid = {'C': np.logspace(-3, 3, 7), 'penalty': ['l1', 'l2']}
knn_param_grid = {'n_neighbors': np.linspace(1, 19, 10, dtype=int).tolist(), 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']}
classifier_param = [dt_param_grid, svc_param_grid, rf_param_grid, logreg_param_grid, knn_param_grid]
cvresults = []
bestEstimator = []
for i in range(len(classifier)):
    clf = GridSearchCV(classifier[i], param_grid=classifier_param[i], cv=StratifiedKFold(n_splits=10), scoring='accuracy', n_jobs=-1, verbose=1)