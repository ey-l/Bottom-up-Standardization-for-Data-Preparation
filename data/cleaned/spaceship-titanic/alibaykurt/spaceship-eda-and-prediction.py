from unicodedata import category
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train_data = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_data = pd.read_csv('data/input/spaceship-titanic/test.csv')
train_data.info()
train_data.describe()
train_data = train_data.drop('Name', axis=1)
test_data = test_data.drop('Name', axis=1)
p_id = test_data['PassengerId']

def bar_plot(variable):
    feature = train_data[variable]
    featureValue = feature.value_counts()
    plt.bar(featureValue.index, featureValue)
    plt.xticks(featureValue.index, featureValue.index.values)
    plt.ylabel('Frequency')
    plt.title(variable)

    plt.figure(figsize=(6, 3))
category1 = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Transported']
for i in category1:
    bar_plot(i)
train_data['HomePlanet'] = train_data['HomePlanet'].fillna('Earth')
test_data['HomePlanet'] = test_data['HomePlanet'].fillna('Earth')
train_data['CryoSleep'] = train_data['CryoSleep'].fillna(False)
test_data['CryoSleep'] = test_data['CryoSleep'].fillna(False)
cabintemp1 = train_data['Cabin'].str.split(pat='/', expand=True)
train_data[['deck', 'temp', 'side']] = cabintemp1
cabintemp2 = test_data['Cabin'].str.split(pat='/', expand=True)
test_data[['deck', 'temp', 'side']] = cabintemp2
train_data['deck'] = train_data['deck'].fillna('F')
train_data['side'] = train_data['side'].fillna('S')
test_data['deck'] = test_data['deck'].fillna('F')
test_data['side'] = test_data['side'].fillna('S')
train_data['Destination'] = train_data['Destination'].fillna('TRAPPIST-1e')
test_data['Destination'] = test_data['Destination'].fillna('TRAPPIST-1e')
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())
test_data['Age'] = test_data['Age'].fillna(train_data['Age'].mean())
train_data['VIP'] = train_data['VIP'].fillna(False)
test_data['VIP'] = test_data['VIP'].fillna(False)
train_data[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = train_data[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(0.0)
test_data[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = test_data[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(0.0)
train_data = train_data.drop(['Cabin', 'temp', 'PassengerId'], axis=1)
test_data = test_data.drop(['Cabin', 'temp', 'PassengerId'], axis=1)
train_data = pd.get_dummies(train_data, columns=['HomePlanet'])
test_data = pd.get_dummies(test_data, columns=['HomePlanet'])
train_data = pd.get_dummies(train_data, columns=['Destination'])
test_data = pd.get_dummies(test_data, columns=['Destination'])
train_data = pd.get_dummies(train_data, columns=['deck'])
test_data = pd.get_dummies(test_data, columns=['deck'])
train_data = pd.get_dummies(train_data, columns=['side'])
test_data = pd.get_dummies(test_data, columns=['side'])
train_data.replace({False: 0, True: 1}, inplace=True)
test_data.replace({False: 0, True: 1}, inplace=True)
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
y = train_data['Transported']
x = train_data.drop(['Transported'], axis=1)
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