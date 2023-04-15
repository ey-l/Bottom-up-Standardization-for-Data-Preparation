import pandas as pd
import numpy as np

def printline(x):
    return pd.DataFrame(x).T
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
df.head(10)
printline(df.isnull().sum())

def rem_num_null(dataset, col):
    for f in col:
        dataset[f] = dataset[f].fillna(dataset[f].mean())
    return dataset

def rem_cat_null(dataset, col):
    for f in col:
        dataset[f] = dataset[f].fillna(dataset[f].mode()[0])
    return dataset
numeric_features = [f for f in df.columns if df[f].dtypes != 'O' and df[f].dtypes != 'bool']
categorical_features = [f for f in df.columns if df[f].dtypes == 'O']
df = rem_num_null(df.copy(), numeric_features)
df = rem_cat_null(df.copy(), categorical_features)
printline(df.isnull().sum())
df.info()
printline(df.columns)
df['Age'].hist()
df['Age'] = df['Age'].astype(int)
df['Face'] = 'norma'
df.loc[df['Age'] < 10, 'Face'] = 'beauty'
df.loc[df['Age'] > 50, 'Face'] = 'ugly'
df['HomePlanet'].hist()
df.loc[df['HomePlanet'] == 'Europa', 'Face'] = 'beauty'
df.loc[df['HomePlanet'] == 'Earth', 'Face'] = 'norma'
df.loc[df['HomePlanet'] == 'Mars', 'Face'] = 'ugly'
num = np.random.randint(0, 150, size=1)
data = np.random.randint(0, 80, size=num[0])
for i in data:
    df.loc[df['Age'] == i, 'Face'] = 'beauty'
num = np.random.randint(0, 100, size=1)
data = np.random.randint(0, 80, size=num[0])
for i in data:
    df.loc[df['Age'] == i, 'Face'] = 'norma'
num = np.random.randint(0, 50, size=1)
data = np.random.randint(0, 80, size=num[0])
for i in data:
    df.loc[df['Age'] == i, 'Face'] = 'ugly'
df
df = df.drop(['Name', 'Cabin', 'PassengerId'], axis=1)
new = df.sample(1000)
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