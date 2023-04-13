import pandas as pd
import numpy as np

def printline(x):
    return pd.DataFrame(x).T
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head(10)
printline(_input1.isnull().sum())

def rem_num_null(dataset, col):
    for f in col:
        dataset[f] = dataset[f].fillna(dataset[f].mean())
    return dataset

def rem_cat_null(dataset, col):
    for f in col:
        dataset[f] = dataset[f].fillna(dataset[f].mode()[0])
    return dataset
numeric_features = [f for f in _input1.columns if _input1[f].dtypes != 'O' and _input1[f].dtypes != 'bool']
categorical_features = [f for f in _input1.columns if _input1[f].dtypes == 'O']
_input1 = rem_num_null(_input1.copy(), numeric_features)
_input1 = rem_cat_null(_input1.copy(), categorical_features)
printline(_input1.isnull().sum())
_input1.info()
printline(_input1.columns)
_input1['Age'].hist()
_input1['Age'] = _input1['Age'].astype(int)
_input1['Face'] = 'norma'
_input1.loc[_input1['Age'] < 10, 'Face'] = 'beauty'
_input1.loc[_input1['Age'] > 50, 'Face'] = 'ugly'
_input1['HomePlanet'].hist()
_input1.loc[_input1['HomePlanet'] == 'Europa', 'Face'] = 'beauty'
_input1.loc[_input1['HomePlanet'] == 'Earth', 'Face'] = 'norma'
_input1.loc[_input1['HomePlanet'] == 'Mars', 'Face'] = 'ugly'
num = np.random.randint(0, 150, size=1)
data = np.random.randint(0, 80, size=num[0])
for i in data:
    _input1.loc[_input1['Age'] == i, 'Face'] = 'beauty'
num = np.random.randint(0, 100, size=1)
data = np.random.randint(0, 80, size=num[0])
for i in data:
    _input1.loc[_input1['Age'] == i, 'Face'] = 'norma'
num = np.random.randint(0, 50, size=1)
data = np.random.randint(0, 80, size=num[0])
for i in data:
    _input1.loc[_input1['Age'] == i, 'Face'] = 'ugly'
_input1
_input1 = _input1.drop(['Name', 'Cabin', 'PassengerId'], axis=1)
new = _input1.sample(1000)
new.info()
new['Face'].hist()
from sklearn.model_selection import train_test_split
y = new['Face']
X = new.drop('Face', axis=1)
X
printline(y)
numeric_features = [f for f in X.columns if X[f].dtypes != 'O' and X[f].dtypes != 'bool']
categorical_features = [f for f in X.columns if X[f].dtypes == 'O']
X = pd.get_dummies(X, categorical_features)
X
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=42)
X_train
X_test
y_train
y_test
from sklearn import tree
clf = tree.DecisionTreeClassifier()